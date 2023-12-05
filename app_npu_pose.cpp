/* FreeRTOS kernel includes. */
#include "FreeRTOS.h"
#include "semphr.h"
#include "task.h"

#include "alg_module.h"
#include "npu_init.h"
#include "alg_conf.h"
#include "app_npu.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "alg_perf.h"
#include "nv12_ops.h"
#include "math.h"
#include "flash.h"
#include <string>
#include <queue>

using namespace cv;

#define GPIO_CTRL_PIN         GPIO_PIN1
#define GPIO_CTRL_PIN_GP      SYS_IP_GPIO3
#define GPIO_CTRL_PIN_PAD     PAD_SDIO0_DATA3
#define GPIO_CTRL_PIN_FUNC    PAD_SDIO0_DATA3_FUNC_GPIO3_1
#define NPU_OSD_BOX_MAX 20
#define POSE_MODEL_FLASH_OFF 0xba0000 
#define POSE_MODEL_SIZE 3283934
#define LABEL_IOU_THRESHOLD (0.9f)

struct npu_osd_info {
    SemaphoreHandle_t lock;

    int det_num;
    int valid;
    struct pose_yolov5_bbox_kp_t bbox_kp[NPU_OSD_BOX_MAX];
};

typedef struct {
   struct npu_osd_info osd_info[2];

   struct alg_ops *pose_yolov5_ops;
   void *pose_yolov5_hnd;

   int npu_ready;
   int busy;

   int osd_idx;
   QueueHandle_t task_queue;

   int skip_cnt;
   int block_cnt;

   GPIOHnd_t gpioHnd;

   int input_wt_idx;
   int input_rd_idx;

   ALG_PERF_DEF(perf);
} AppMgr_t;

static float box_iou(cv::Rect rect1, cv::Rect rect2)
{
    int x1 = MAX(rect1.x, rect2.x);
    int y1 = MAX(rect1.y, rect2.y);
    int x2 = MIN(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = MIN(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x1 >= x2 || y1 >= y2)
        return 0.0f;

    int inner = (x2 - x1) * (y2 - y1);
    int total = rect1.width * rect1.height + rect2.width * rect2.height - inner;

    return (1.0f * inner) / total;
}

static void get_rect_from_bbox(cv::Rect& rect, struct pose_yolov5_bbox_kp_t& bbox_kp) {
    rect.x = (int)(bbox_kp.bbox.center_w - bbox_kp.bbox.len_w / 2.0f);
    rect.y = (int)(bbox_kp.bbox.center_h - bbox_kp.bbox.len_h / 2.0f);
    rect.width = (int)(bbox_kp.bbox.len_w);
    rect.height = (int)(bbox_kp.bbox.len_h);
}

class PoseQueueSmooth {
private:
    cv::Rect Rect_total;
    struct pose_yolov5_keypoint_t Keypoint_total[17];
public:
    bool flag_in;
    std::queue<struct pose_yolov5_bbox_kp_t> queue;
    PoseQueueSmooth() {
        flag_in = false;
        Rect_total.x = 0;
        Rect_total.y = 0;
        Rect_total.height = 0;
        Rect_total.width = 0;
        for (int i = 0; i < 17; i++) {
            Keypoint_total[i].confidence = 0.0f;
            Keypoint_total[i].h = 0.0f;
            Keypoint_total[i].w = 0.0f;
        }
    }
    void queueSmoothClear();
    void bboxkpSmooth(struct pose_yolov5_bbox_kp_t& bbox_kp, cv::Rect& rect_rtn);
};

void PoseQueueSmooth::queueSmoothClear() {
    while (!queue.empty()) {
        queue.pop();
    }
    flag_in = false;
    Rect_total.x = 0;
    Rect_total.y = 0;
    Rect_total.width = 0;
    Rect_total.height = 0;
    for (int i = 0; i < 17; i++) {
        Keypoint_total[i].confidence = 0.0f;
        Keypoint_total[i].h = 0.0f;
        Keypoint_total[i].w = 0.0f;
    }
}

void PoseQueueSmooth::bboxkpSmooth(struct pose_yolov5_bbox_kp_t& bbox_kp, cv::Rect& rect_rtn) {
    if (queue.size() == 0) {
        cv::Rect rect;
        get_rect_from_bbox(rect, bbox_kp);
        queue.push(bbox_kp);
        Rect_total.x += rect.x;
        Rect_total.y += rect.y;
        Rect_total.width += rect.width;
        Rect_total.height += rect.height;
        for (int i = 0; i < 17; i++) {
            Keypoint_total[i].confidence += bbox_kp.kp[i].confidence;
            Keypoint_total[i].h += bbox_kp.kp[i].h;
            Keypoint_total[i].w += bbox_kp.kp[i].w;
        }
        flag_in = true;
    } else {
        cv::Rect tmprect, rect;
        get_rect_from_bbox(tmprect, queue.front());
        get_rect_from_bbox(rect, bbox_kp);
        if (queue.size() < 8) {
            queue.push(bbox_kp);
            Rect_total.x += rect.x;
            Rect_total.y += rect.y;
            Rect_total.width += rect.width;
            Rect_total.height += rect.height;
            for (int i = 0; i < 17; i++) {
                Keypoint_total[i].confidence += bbox_kp.kp[i].confidence;
                Keypoint_total[i].h += bbox_kp.kp[i].h;
                Keypoint_total[i].w += bbox_kp.kp[i].w;
            }
        } else {
            Rect_total.x -= tmprect.x;
            Rect_total.y -= tmprect.y;
            Rect_total.width -= tmprect.width;
            Rect_total.height -= tmprect.height;
            for (int i = 0; i < 17; i++) {
                Keypoint_total[i].confidence -= queue.front().kp[i].confidence;
                Keypoint_total[i].h -= queue.front().kp[i].h;
                Keypoint_total[i].w -= queue.front().kp[i].w;
            }
            queue.pop();
            Rect_total.x += rect.x;
            Rect_total.y += rect.y;
            Rect_total.width += rect.width;
            Rect_total.height += rect.height;
            for (int i = 0; i < 17; i++) {
                Keypoint_total[i].confidence += bbox_kp.kp[i].confidence;
                Keypoint_total[i].h += bbox_kp.kp[i].h;
                Keypoint_total[i].w += bbox_kp.kp[i].w;
            }
            queue.push(bbox_kp);
        }
        rect_rtn.x = Rect_total.x / queue.size();
        rect_rtn.y = Rect_total.y / queue.size();
        rect_rtn.width = Rect_total.width / queue.size();
        rect_rtn.height = Rect_total.height / queue.size();
        for (int i = 0; i < 17; i++) {
            bbox_kp.kp[i].confidence = Keypoint_total[i].confidence / queue.size();
            bbox_kp.kp[i].h = Keypoint_total[i].h / queue.size();
            bbox_kp.kp[i].w = Keypoint_total[i].w / queue.size();
        }
        flag_in = true;
    }
}     

static AppMgr_t app_mgr;
static PoseQueueSmooth queue_smooth[5];

static void reset_all_queue_flag() {
    for (int i = 0; i < 5; i++) {
        queue_smooth[i].flag_in = false;
    }
}

static void process_rect_smooth(struct pose_yolov5_bbox_kp_t& bbox_kp, cv::Rect& rect_rtn) {
    int i = 0;
    float max_iou = 0.0f;
    float tmp_iou = 0.0f;
    bool find_closest_iou = false;
    cv::Rect rect;
    get_rect_from_bbox(rect, bbox_kp);
    for (int j = 0; j < 5; j++) {
        if (queue_smooth[j].queue.size() != 0) {
            cv::Rect tmprect;
            get_rect_from_bbox(tmprect, queue_smooth[j].queue.back());
            tmp_iou = box_iou(rect, tmprect);
            if (tmp_iou >= LABEL_IOU_THRESHOLD && tmp_iou > max_iou) {
                max_iou = tmp_iou;
                i = j;
                find_closest_iou = true;
            }
        }
        if (!find_closest_iou) {
            for (int j = 0; i < 5; j++) {
                if (queue_smooth[j].queue.size() == 0) {
                    i = j;
                    find_closest_iou = true;
                    break;
                }
            }
        }
    }
    if (find_closest_iou) {
        queue_smooth[i].bboxkpSmooth(bbox_kp, rect_rtn);
    }
}

static void clear_queue_name_not_in() {
    for (int i = 0; i < 5; i++) {
        if (queue_smooth[i].flag_in == false) {
            queue_smooth[i].queueSmoothClear();
        }
    }
}

#define MAX_BUF_CNT 3

static uint8_t pose_yolov5_model_input[MAX_BUF_CNT][POSE_INPUT_WIDTH * POSE_INPUT_HEIGHT * 4] __attribute__((aligned(64)));

extern DMA_UNCACHEBLE_DATA uint8_t aidemo_model_weight[2 * 1024 * 1024] __attribute__((aligned(64)));

static int skeleton[][2] = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
    {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3},
    {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}
};

static cv::Scalar palette[] = {
    cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51),
    cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0),
    cv::Scalar(255, 153, 255), cv::Scalar(153, 204, 255),
    cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255),
    cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
    cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102),
    cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153),
    cv::Scalar(102, 255, 102), cv::Scalar(51, 255, 51),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
    cv::Scalar(255, 0, 0),cv::Scalar(255, 255, 255)
};

static int pose_limb_color_idx[] = {9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};
static int pose_kpt_color_idx[] = {16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};

int app_pose_npu_is_busy(void)
{
    return app_mgr.busy;
}

int app_pose_npu_proc(struct npu_proc_param *param)
{
    if (!app_mgr.npu_ready)
        return 0;

    HAL_QUEU_PUT(app_mgr.task_queue, param, 0);
    app_mgr.busy = 1;
    return 0;
}

int app_pose_npu_osd_handle(void *y, void *uv, int width, int height)
{
    int last_idx = !app_mgr.osd_idx;
    struct npu_osd_info *osd_info = &app_mgr.osd_info[last_idx];

    if (!app_mgr.npu_ready)
        return 0;

    HAL_LOCK(osd_info->lock);
    if (!osd_info->valid) {
        HAL_UNLOCK(osd_info->lock);
        return 0;
    }

    int gpio_level = HAL_GPIO_GetPin(app_mgr.gpioHnd, GPIO_CTRL_PIN);
    if (!gpio_level) {
        app_mgr.skip_cnt = 1;
        HAL_UNLOCK(osd_info->lock);
        return 0;
    }
    app_mgr.skip_cnt = 0;

#if 0
    cv::Scalar color(255, 0, 0);
    cv::Rect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = 100;
    rect.height = 100;
    nv12_rectangle(y, uv, width, height, rect, color, -1);
#endif

    for (int i = 0; i < osd_info->det_num; i++) {
        struct pose_yolov5_bbox_kp_t tmp_bbox_kp;
        struct pose_yolov5_bbox_kp_t *bbox_kp = &tmp_bbox_kp;

        memcpy(bbox_kp, &osd_info->bbox_kp[i], sizeof(struct pose_yolov5_bbox_kp_t));

        PoseYolov5ScaleArg_t pose_yolov5_scale_param;
        pose_yolov5_scale_param.bbox_kp = bbox_kp;
        pose_yolov5_scale_param.bbox_kp_cnt = 1;
        pose_yolov5_scale_param.src_h = pose_yolov5_conf.post_conf->det_h;
        pose_yolov5_scale_param.src_w = pose_yolov5_conf.post_conf->det_w;
        pose_yolov5_scale_param.dst_h = height;
        pose_yolov5_scale_param.dst_w = width;
        ALG_CTRL(app_mgr.pose_yolov5_ops, app_mgr.pose_yolov5_hnd, ALG_MODEL_CMD_SCALE,
                &pose_yolov5_scale_param);

        cv::Rect rect;
        get_rect_from_bbox(rect, tmp_bbox_kp);
        // rect.x = (int)(bbox_kp->bbox.center_w - bbox_kp->bbox.len_w / 2.0f);
        // rect.y = (int)(bbox_kp->bbox.center_h - bbox_kp->bbox.len_h / 2.0f);
        // rect.width = (int)(bbox_kp->bbox.len_w);
        // rect.height = (int)(bbox_kp->bbox.len_h);

        /* Rect and keypoints smooth process */
        reset_all_queue_flag();
        process_rect_smooth(tmp_bbox_kp, rect);
        clear_queue_name_not_in();
        
        nv12_rectangle(y, uv, width, height, rect, palette[0], 2);

        cv::Point text_point;
        if (rect.x > width - 200) {
            text_point.x = width - 200;
        } else if (rect.x < 10) {
            text_point.x = 10;
        } else {
            text_point.x = rect.x;
        }
        if (rect.y < 50) {
            text_point.y = 50;
        } else if (rect.y > height - 10) {
            text_point.y = height - 10;
        } else {
            text_point.y = rect.y;
        }
 
        cv::Scalar color(0, 255, 0);
        const char *action = "normal";

        /* Falling detect */
#if 0
        float delta_h = abs(bbox_kp->kp[16/3].h + bbox_kp->kp[19/3].h -
                    bbox_kp->kp[34/3].h -  bbox_kp->kp[37/3].h) / 2;
        if (delta_h < 140 && bbox_kp->kp[17 / 3].confidence > 0.5f &&
                bbox_kp->kp[35 / 3].confidence > 0.5f) {
            action = "falling";
            color = cv::Scalar(255, 0, 0);
        }
#else
#define BOX_KP_X(bbox_kp, n) ((bbox_kp)->kp[(n) / 3].w)
#define BOX_KP_Y(bbox_kp, n) ((bbox_kp)->kp[(n) / 3].h)
#define BOX_KP_C(bbox_kp, n) ((bbox_kp)->kp[(n) / 3].confidence)

        int box_x[7];
        int box_y[7];
        float slope[4];
        float min_slope1;
        float min_slope2;
        int box_valid[7];
        int box_gp[8] = {15, 18, 33, 36, 39, 42, 45, 48};

        memset(box_valid, 0, sizeof(box_valid));
        for (int i = 0; i < ARRAY_SIZE(box_gp); i++) {
            if (i == 0) {
                if (BOX_KP_C(bbox_kp, box_gp[i]) > 0.5f &&
                        BOX_KP_C(bbox_kp, box_gp[i + 1]) > 0.5f) {
                    box_x[i] = (int)((BOX_KP_X(bbox_kp, box_gp[i]) +
                            BOX_KP_X(bbox_kp, box_gp[i])) / 2);
                    box_y[i] = (int)((BOX_KP_Y(bbox_kp, box_gp[i]) +
                            BOX_KP_Y(bbox_kp, box_gp[i])) / 2);
                    box_valid[i] = 1;
                } else
                    box_valid[i] = 0;
                i++;
            } else {
                if (BOX_KP_C(bbox_kp, box_gp[i]) > 0.5f) {
                    box_x[i - 1] = (int)BOX_KP_X(bbox_kp, box_gp[i]);
                    box_y[i - 1] = (int)BOX_KP_Y(bbox_kp, box_gp[i]);
                    box_valid[i - 1] = 1;
                } else
                    box_valid[i - 1] = 0;
            }
        }

        for (int i = 0; i < ARRAY_SIZE(slope); i++)
            slope[i] = 1.0f;

        if (box_valid[0] && box_valid[1] && box_x[0] != box_x[1])
            slope[0] = fabsf((float)(box_y[0] - box_y[1]) / (box_x[0] - box_x[1]));

        if (box_valid[0] && box_valid[2] && box_x[0] != box_x[2])
            slope[1] = fabsf((float)(box_y[0] - box_y[2]) / (box_x[0] - box_x[2]));

        if (box_valid[3] && box_valid[5] && box_x[3] != box_x[5])
            slope[2] = fabsf((float)(box_y[3] - box_y[5]) / (box_x[3] - box_x[5]));

        if (box_valid[4] && box_valid[6] && box_x[4] != box_x[6])
            slope[3] = fabsf((float)(box_y[4] - box_y[6]) / (box_x[4] - box_x[6]));

        // min_slope = slope[0];
        // for (int i = 1; i < ARRAY_SIZE(slope); i++) {
        //     if (min_slope > slope[i])
        //         min_slope = slope[i];
        // }
        min_slope1 = (slope[0] < slope[1] ? slope[0] : slope[1]);
        min_slope2 = (slope[2] < slope[3] ? slope[2] : slope[3]);


        if ((min_slope1 < 1.0f) && (min_slope2 < 0.5f)) {
            action = "falling";
            color = cv::Scalar(255, 0, 0);
            //ts_printf("slope: (%f,%f,%f,%f)\n",
            //        slope[0], slope[1], slope[2], slope[3]);
            //for (int i = 0; i < ARRAY_SIZE(box_x); i++) {
            //    ts_printf("box[%d]: (%d, %d, %d)\n",
            //            i, box_valid[i], box_x[i], box_y[i]);
            //}
        }
#endif

        char label[64];
        snprintf(label, 63, "%s %d%%", action, (int)(100.0f * bbox_kp->bbox.box_confidence + 0.5f));
        nv12_putText(y, uv, width, height, label,
                text_point, FONT_HERSHEY_SIMPLEX, 1.3f, color, 2);

        for (int j = 0; j < 17; j++) {
            if (bbox_kp->kp[j].confidence < 0.5f) {
                continue;
            }
            cv::Point point;
            point.x = (int)(bbox_kp->kp[j].w);
            point.y = (int)(bbox_kp->kp[j].h);
            nv12_circle(y, uv, width, height, point, 6, palette[pose_kpt_color_idx[j]], 4);
        }

        for (int j = 0; j < sizeof(skeleton) / (2 * sizeof(int)); j++) {
            if ((bbox_kp->kp[skeleton[j][0] - 1].confidence < 0.5f) || 
                    (bbox_kp->kp[skeleton[j][1] - 1].confidence < 0.5f)) {
                continue;
            }
            cv::Point start, end;
            start.x = (int)(bbox_kp->kp[skeleton[j][0] - 1].w);
            start.y = (int)(bbox_kp->kp[skeleton[j][0] - 1].h);
            end.x = (int)(bbox_kp->kp[skeleton[j][1] - 1].w);
            end.y = (int)(bbox_kp->kp[skeleton[j][1] - 1].h);
            nv12_line(y, uv, width, height, start, end, palette[pose_limb_color_idx[j]], 6);
        }
    }
    HAL_UNLOCK(osd_info->lock);

    return 0;
}

static void app_pre_task(void *param)
{
    while (1) {
        struct npu_proc_param app_proc_param;

        if (!app_mgr.pose_yolov5_hnd || app_mgr.skip_cnt) {
            vTaskDelay(2);
            continue;
        }
        app_mgr.block_cnt = 0;
        if (HAL_QUEU_GET(app_mgr.task_queue, &app_proc_param, 2) != pdPASS) {
            if (!app_mgr.skip_cnt)
                app_mgr.busy = 0;
            continue;
        }

        int next_idx = (app_mgr.input_wt_idx + 1) % MAX_BUF_CNT;
        if (app_mgr.input_rd_idx == next_idx)
            continue;

        PoseYolov5PreArg_t pose_yolov5_pre_param;
        pose_yolov5_pre_param.in = app_proc_param.scale_image;
        pose_yolov5_pre_param.out = pose_yolov5_model_input[app_mgr.input_wt_idx];
        ALG_CTRL(app_mgr.pose_yolov5_ops, app_mgr.pose_yolov5_hnd, ALG_MODEL_CMD_PRE, &pose_yolov5_pre_param);
        ALG_CACHE_CLEAN(pose_yolov5_model_input[app_mgr.input_wt_idx], sizeof(pose_yolov5_model_input));
        app_mgr.input_wt_idx = next_idx;
    }

    vTaskDelete(NULL);
}

static void app_task(void *param)
{
    int ret;

    npu_init();

    app_mgr.pose_yolov5_ops = ALG_GET_OPS(ALG_MODEL_POSE_YOLOV5);
    HAL_SANITY_CHECK(NULL != app_mgr.pose_yolov5_ops);

    pose_yolov5_conf.heap_addr = ddr_malloc(pose_yolov5_conf.heap_size);
    HAL_SANITY_CHECK(NULL != pose_yolov5_conf.heap_addr);

    FlashHnd_t flashhnd = Flash_Init();
    HAL_SANITY_CHECK(flashhnd != NULL);

    if (HAL_OK != Flash_Read(flashhnd, POSE_MODEL_FLASH_OFF, (uint8_t *)(&aidemo_model_weight), POSE_MODEL_SIZE, 1000)) {
        ts_printf("pose_app_task: flash Read weight FAILED!\n");
    }
    pose_yolov5_conf.model = aidemo_model_weight;
    pose_yolov5_conf.model_size = POSE_MODEL_SIZE;

    app_mgr.pose_yolov5_hnd = ALG_INIT(app_mgr.pose_yolov5_ops, &pose_yolov5_conf);
    HAL_SANITY_CHECK(NULL != app_mgr.pose_yolov5_hnd);

    ts_printf("%s init done, CPU_MHZ:%d\n", __func__, CPU_MHZ);

    app_mgr.npu_ready = 1;

    ALG_PERF_INIT(app_mgr.perf, "npu_perf", 50);
    while (1) {
        if (app_mgr.input_rd_idx == app_mgr.input_wt_idx) {
            vTaskDelay(1);
            continue;
        }

        ALG_PERF_END(app_mgr.perf);
        ALG_PERF_START(app_mgr.perf);

        PoseYolov5ProcArg_t pose_yolov5_proc_param;
        void *in[1];
        in[0] = pose_yolov5_model_input[app_mgr.input_rd_idx];
        pose_yolov5_proc_param.in = in;
        pose_yolov5_proc_param.out = NULL;
        ret = ALG_PROC(app_mgr.pose_yolov5_ops, app_mgr.pose_yolov5_hnd, &pose_yolov5_proc_param);
        HAL_SANITY_CHECK(ret == 0);

        PoseYolov5PostArg_t pose_yolov5_post_param;
        pose_yolov5_post_param.model_output = NULL;
        pose_yolov5_post_param.det_num = 0;
        pose_yolov5_post_param.bbox_kp = NULL;
        ret = ALG_CTRL(app_mgr.pose_yolov5_ops, app_mgr.pose_yolov5_hnd, ALG_MODEL_CMD_POST, &pose_yolov5_post_param);
        //ts_printf("%s det_num:%d\n", __func__, pose_yolov5_post_param.det_num);

        if (pose_yolov5_post_param.det_num > NPU_OSD_BOX_MAX)
            pose_yolov5_post_param.det_num = NPU_OSD_BOX_MAX;

        struct npu_osd_info *osd_info = &app_mgr.osd_info[app_mgr.osd_idx];

        HAL_LOCK(osd_info->lock);
        osd_info->det_num = pose_yolov5_post_param.det_num;
        memcpy(&osd_info->bbox_kp[0], pose_yolov5_post_param.bbox_kp, 
                pose_yolov5_post_param.det_num * sizeof(struct pose_yolov5_bbox_kp_t));

        osd_info->valid = 1;
        app_mgr.osd_idx = !app_mgr.osd_idx;
        HAL_UNLOCK(osd_info->lock);

        app_mgr.input_rd_idx = (app_mgr.input_rd_idx + 1) % MAX_BUF_CNT;
    }

    vTaskDelete(NULL);
}

static void app_npu_ctrl_init(void)
{
    PinmuxConf_t conf = PINMUX_DEFAULT_CONF();
    conf.pull         = PAD_PULL_UP;
    conf.funcid       = GPIO_CTRL_PIN_FUNC;
    sys_pinmux_conf(GPIO_CTRL_PIN_PAD, &conf);

    app_mgr.gpioHnd = HAL_GPIO_Init(GPIO_CTRL_PIN_GP);
    HAL_SANITY_CHECK(NULL != app_mgr.gpioHnd);

    GPIOConf_t gpioConf = GPIO_DEFAULT_CONFIG();
    gpioConf.dir        = GPIO_DIR_IN;
    gpioConf.pin        = GPIO_CTRL_PIN;
    HAL_GPIO_ConfigPin(app_mgr.gpioHnd, &gpioConf);
}

int app_npu_pose_init(void)
{
    memset(&app_mgr, 0, sizeof(AppMgr_t));

    alg_init();

    app_npu_ctrl_init();

    for (int i = 0; i < ARRAY_SIZE(app_mgr.osd_info); i++) {
        HAL_LOCK_INIT(app_mgr.osd_info[i].lock);
    }

    app_mgr.task_queue = xQueueCreate(1, sizeof(struct npu_proc_param));
    HAL_SANITY_CHECK(NULL != app_mgr.task_queue);

    BaseType_t xTask = xTaskCreate(app_task, "app_task", 2 * 1024, NULL, tskIDLE_PRIORITY + 1, NULL);
    if (xTask != pdPASS) {
        ts_printf("app task create fail!\n");
        return -1;
    }

    xTask = xTaskCreate(app_pre_task, "app_pre_task", 2 * 1024, NULL, tskIDLE_PRIORITY + 1, NULL);
    if (xTask != pdPASS) {
        ts_printf("app pre task create fail!\n");
        return -1;
    }

    return 0;
}

extern int _npu_busy;
void app_pose_npu_monitor(void)
{
    if (!app_mgr.npu_ready)
        return;

    if (_npu_busy) {
        app_mgr.block_cnt++;
    }

    if (app_mgr.block_cnt > 3) {
        npu_monitor();
    } else {
        ts_printf("npu work\n");
    }
}
