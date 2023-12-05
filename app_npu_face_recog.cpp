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
#include "alg_mem_pool.h"
#include "alg_list.h"
#include "flash.h"
#include <string>
#include <queue>

using namespace cv;

#define GPIO_CTRL_PIN         GPIO_PIN1
#define GPIO_CTRL_PIN_GP      SYS_IP_GPIO3
#define GPIO_CTRL_PIN_PAD     PAD_SDIO0_DATA3
#define GPIO_CTRL_PIN_FUNC    PAD_SDIO0_DATA3_FUNC_GPIO3_1

#define NPU_OSD_BOX_MAX 20

struct npu_osd_info {
    SemaphoreHandle_t lock;

    int det_num;
    int valid;

    struct retinaface_feature_t feature[NPU_OSD_BOX_MAX];
    float min_val[NPU_OSD_BOX_MAX];
    int min_idx[NPU_OSD_BOX_MAX];
};

#define MAX_LABEL_CNT 16
#define MAX_RECT_CNT  16

#define RETINAFACE_MODEL_FLASH_OFF 0x630000 
#define RETINAFACE_MODEL_SIZE 523564
#define MOBILEFACENET_MODEL_FLASH_OFF 0x500000
#define MOBILEFACENET_MODEL_SIZE 1211664
#define LABEL_IOU_THRESHOLD (0.9f)

struct rect_label {
    struct list_head list;
    cv::Rect rect;
    const char *label[MAX_LABEL_CNT];
    int wt_idx;
    int rd_idx;
};

typedef struct {
    struct npu_osd_info osd_info[2];

    struct alg_ops *retinaface_ops;
    void *retinaface_hnd;

    struct alg_ops *mobilefacenet_ops;
    void *mobilefacenet_hnd;

   int npu_ready;
    int busy;

    int osd_idx;
    QueueHandle_t task_queue;

    int skip_cnt;
    int block_cnt;

    struct alg_mem_pool *rect_mem_pool;
    struct list_head rect_head;

    GPIOHnd_t gpioHnd;

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

class FaceQueueSmooth {
private:
    std::queue<cv::Rect> queue;
    cv::Rect Rect_total;
public:
    std::string name;
    bool flag_in;
    FaceQueueSmooth() {
        flag_in = false;
        name = "";
        Rect_total.x = 0;
        Rect_total.y = 0;
        Rect_total.height = 0;
        Rect_total.width = 0;
    }
    void queueSmoothClear();
    void rectSmooth(cv::Rect& rect, std::string& label_name);
};

void FaceQueueSmooth::queueSmoothClear() {
    while (!queue.empty()) {
        queue.pop();
    }
    flag_in = false;
    name = "";
    Rect_total.x = 0;
    Rect_total.y = 0;
    Rect_total.width = 0;
    Rect_total.height = 0;
}

void FaceQueueSmooth::rectSmooth(cv::Rect& rect, std::string& label_name) {
    if (queue.size() == 0) {
        name = label_name;
        queue.push(rect);
        Rect_total.x += rect.x;
        Rect_total.y += rect.y;
        Rect_total.width += rect.width;
        Rect_total.height += rect.height;
        flag_in = true;
    } else {
        cv::Rect tmprect = queue.back();
        float iou_for_smooth = box_iou(rect, tmprect);
        if (iou_for_smooth < LABEL_IOU_THRESHOLD) {
            queueSmoothClear();
            name = label_name;
            queue.push(rect);
            Rect_total.x += rect.x;
            Rect_total.y += rect.y;
            Rect_total.width += rect.width;
            Rect_total.height += rect.height;
            flag_in = true;
        } else {
            if (queue.size() < 6) {
                queue.push(rect);
                Rect_total.x += rect.x;
                Rect_total.y += rect.y;
                Rect_total.width += rect.width;
                Rect_total.height += rect.height;
            } else {
                Rect_total.x -= queue.front().x;
                Rect_total.y -= queue.front().y;
                Rect_total.width -= queue.front().width;
                Rect_total.height -= queue.front().height;
                queue.pop();
                Rect_total.x += rect.x;
                Rect_total.y += rect.y;
                Rect_total.width += rect.width;
                Rect_total.height += rect.height;
                queue.push(rect);
            }
            rect.x = Rect_total.x / queue.size();
            rect.y = Rect_total.y / queue.size();
            rect.width = Rect_total.width / queue.size();
            rect.height = Rect_total.height / queue.size();
            flag_in = true;
        }
    }
}     

static AppMgr_t app_mgr;
static FaceQueueSmooth queue_smooth[5];

static void reset_all_queue_flag() {
    for (int i = 0; i < 5; i++) {
        queue_smooth[i].flag_in = false;
    }
}

static void process_rect_and_name(cv::Rect& rect, std::string& name) {
    int i = 0;
    bool flag = false;
    for (; i < 5; i++) {
        if ((queue_smooth[i].name == name) || queue_smooth[i].name == "") {
            flag = true;
            break;
        } //TODO
    }
    queue_smooth[i].rectSmooth(rect, name);
}

static void clear_queue_name_not_in() {
    for (int i = 0; i < 5; i++) {
        if (queue_smooth[i].flag_in == false) {
            queue_smooth[i].queueSmoothClear();
        }
    }
}

static uint8_t retinaface_model_input[320*257*4] __attribute__((aligned(64)));

static uint8_t mobilefacenet_model_input[112 * 113 * 4] __attribute__((aligned(64)));

DMA_UNCACHEBLE_DATA uint8_t aidemo_model_weight[3300000] __attribute__((aligned(64)));
DMA_UNCACHEBLE_DATA uint8_t aidemo_model_weight2[524000] __attribute__((aligned(64)));

int app_face_recog_npu_is_busy(void)
{
    return app_mgr.busy;
}

int app_face_recog_npu_proc(struct npu_proc_param *param)
{
    if (!app_mgr.npu_ready)
        return 0;

    HAL_QUEU_PUT(app_mgr.task_queue, param, 0);
    app_mgr.busy = 1;
    return 0;
}

int app_face_recog_npu_osd_handle(void *y, void *uv, int width, int height)
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

    LIST_HEAD(tmp_head);
    for (int i = 0; i < osd_info->det_num; i++) {
        struct retinaface_feature_t tmp_feature;
        struct retinaface_feature_t* feature = &tmp_feature;

        memcpy(&tmp_feature, &osd_info->feature[i], sizeof(struct retinaface_feature_t));
        RetinafaceScaleArg_t retinaface_scale_param;
        retinaface_scale_param.feature = feature;
        retinaface_scale_param.feature_cnt = 1;
        retinaface_scale_param.src_h = retinaface_conf.post_conf->det_h;
        retinaface_scale_param.src_w = retinaface_conf.post_conf->det_w;;
        retinaface_scale_param.dst_h = height;
        retinaface_scale_param.dst_w = width;
        ALG_CTRL(app_mgr.retinaface_ops, app_mgr.retinaface_hnd, ALG_MODEL_CMD_SCALE,
                &retinaface_scale_param);

        const char *name = "Unknown";
        if (osd_info->min_idx[i] != -1)
            name = mobilefacenet_database_name[osd_info->min_idx[i]];

        /* validate box */
        if ((int)feature->bbox.l_w < 0 || (int)feature->bbox.l_h < 0 ||
                (int)feature->bbox.r_w < 0 || (int)feature->bbox.r_h < 0 ||
                feature->bbox.l_w > width || feature->bbox.l_h > height ||
                feature->bbox.r_w > width || feature->bbox.r_h > height ||
                feature->bbox.l_w > feature->bbox.r_w ||
                feature->bbox.l_h > feature->bbox.r_h) {
            ts_printf("box error (width, height):(%d, %d) (l_w, l_h, r_w, r_h): (%.1f, %.1f, %.1f, %.1f)\n",
                    width, height, feature->bbox.l_w, feature->bbox.l_h,
                    feature->bbox.r_w, feature->bbox.r_h);
            continue;
        }

        cv::Rect rect;
        rect.x = (int)(feature->bbox.l_w);
        rect.y = (int)(feature->bbox.l_h);
        rect.width = (int)(feature->bbox.r_w) - (int)(feature->bbox.l_w);
        rect.height = (int)(feature->bbox.r_h) - (int)(feature->bbox.l_h);

#define SAME_PERSON_IOU (0.2f)
#if 1
        struct rect_label *p_rect_label = NULL;
        struct rect_label *nearest_rect_label = NULL;
        float max_iou = 0.0f;
        list_for_each_entry(p_rect_label, &app_mgr.rect_head, list) {
            float iou = box_iou(rect, p_rect_label->rect);
            if (iou > max_iou) {
                max_iou = iou;
                nearest_rect_label = p_rect_label;
            }
        }

        if (max_iou > SAME_PERSON_IOU) {
            list_del(&nearest_rect_label->list);
            list_add_tail(&nearest_rect_label->list, &tmp_head);
        } else {
            list_for_each_entry(p_rect_label, &tmp_head, list) {
                float iou = box_iou(rect, p_rect_label->rect);
                if (iou > max_iou) {
                    max_iou = iou;
                    nearest_rect_label = p_rect_label;
                }
            }

            /* insert new node */
            if (max_iou <= SAME_PERSON_IOU) {
                nearest_rect_label = (struct rect_label *)alg_request_buf(app_mgr.rect_mem_pool);
                HAL_SANITY_CHECK(NULL != nearest_rect_label);

                HAL_MEMSET(nearest_rect_label, 0, sizeof(*nearest_rect_label));
                list_add_tail(&nearest_rect_label->list, &tmp_head);
            }
        }

        HAL_SANITY_CHECK(NULL != nearest_rect_label);
        nearest_rect_label->rect = rect;

        /* name into queue */
        nearest_rect_label->label[nearest_rect_label->wt_idx] = name;
        nearest_rect_label->wt_idx = (nearest_rect_label->wt_idx + 1) % MAX_LABEL_CNT;
        if (nearest_rect_label->wt_idx == nearest_rect_label->rd_idx) {
            nearest_rect_label->rd_idx = (nearest_rect_label->rd_idx + 1) % MAX_LABEL_CNT;
        }

        /* checkout the name, often mode  */
        int name_total = 0;
        int name_cnt[MAX_LABEL_CNT];
        const char *name_label[MAX_LABEL_CNT];
        memset(name_cnt, 0, sizeof(name_cnt));
        memset(name_label, 0, sizeof(name_label));
        for (int idx = nearest_rect_label->rd_idx;
                idx != nearest_rect_label->wt_idx;) {
            const char *label = nearest_rect_label->label[idx];

            int name_idx = 0;
            while (1) {
                if (name_idx < name_total) {
                    if (name_label[name_idx] == label) {
                        name_cnt[name_idx]++;
                        break;
                    }
                } else {
                    name_label[name_idx] = label;
                    name_cnt[name_idx]++;
                    name_total++;
                    break;
                }
                name_idx++;
            }
            idx = (idx + 1) % MAX_LABEL_CNT;
        }

        int max_idx = 0;
        for (int i = 0; i < name_total; i++) {
            if (name_cnt[i] > name_cnt[max_idx]) {
                max_idx = i;
            }
        }
        name = name_label[max_idx];
        if (name == NULL) {
            ts_printf("max_idx:%d name is null\n", max_idx);
        } else {
            //ts_printf("max_idx:%d name:%s\n", max_idx, name);
        }
     
        //rect_smooth process
        reset_all_queue_flag();
        std::string s_name = name;
        process_rect_and_name(rect, s_name);
        clear_queue_name_not_in();
#endif


        /* show box and name */
        cv::Point point;
        if (rect.x > width - 200) {
            point.x = width - 200;
        } else if (rect.x < 10) {
            point.x = 10;
        } else {
            point.x = rect.x;
        }
        if (rect.y < 100) {
            point.y = 100;
        } else {
            point.y = rect.y - 100;
        }
        if (strcmp(name, "Unknown") == 0) {
            nv12_rectangle(y, uv, width, height, rect, cv::Scalar(255, 0, 0), 2);
            char label[64];
            snprintf(label, 63, "box:%dx%d", rect.width, rect.height);
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(255, 0, 0), 2);
            point.y += 50;
            snprintf(label, 63, "face:%d%%", (int)(100.0f * feature->confidence.fore + 0.5f));
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(255, 0, 0), 2);
            point.y += 50;
            snprintf(label, 63, "%s:%.2f", name, osd_info->min_val[i]);
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(255, 0, 0), 2);
        } else {
            nv12_rectangle(y, uv, width, height, rect, cv::Scalar(0, 255, 0), 2);
            char label[64];
            snprintf(label, 63, "box:%dx%d", rect.width, rect.height);
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(0, 255, 0), 2);
            point.y += 50;
            snprintf(label, 63, "face:%d%%", (int)(100.0f * feature->confidence.fore + 0.5f));
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(0, 255, 0), 2);
            point.y += 50;
            snprintf(label, 63, "%s:%.2f", name, osd_info->min_val[i]);
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(0, 255, 0), 2);
        }
    }

    struct rect_label *p_rect_label = NULL;
    struct rect_label *p_next_rect_label = NULL;
    list_for_each_entry_safe(p_rect_label, p_next_rect_label, &app_mgr.rect_head, list) {
        list_del(&p_rect_label->list);
        alg_release_buf(app_mgr.rect_mem_pool, p_rect_label);
    }

    if (!list_empty(&tmp_head)) {
        list_add_tail(&app_mgr.rect_head, &tmp_head);
        list_del(&tmp_head);
    }
    HAL_UNLOCK(osd_info->lock);

    return 0;
}

static void app_task(void *param)
{
    int ret;

    npu_init();

    FlashHnd_t flashhnd = Flash_Init();
    HAL_SANITY_CHECK(flashhnd != NULL);

    app_mgr.retinaface_ops = ALG_GET_OPS(ALG_MODEL_RETINAFACE);
    HAL_SANITY_CHECK(NULL != app_mgr.retinaface_ops);

    retinaface_conf.heap_addr = ddr_malloc(retinaface_conf.heap_size);
    HAL_SANITY_CHECK(NULL != retinaface_conf.heap_addr);

    if (HAL_OK != Flash_Read(flashhnd, RETINAFACE_MODEL_FLASH_OFF, (uint8_t *)(&aidemo_model_weight2), RETINAFACE_MODEL_SIZE, 1000)) {
        ts_printf("retinaface_app_task: flash Read weight FAILED!\n");
    }
    retinaface_conf.model = aidemo_model_weight2;
    retinaface_conf.model_size = RETINAFACE_MODEL_SIZE;

    app_mgr.retinaface_hnd = ALG_INIT(app_mgr.retinaface_ops, &retinaface_conf);        
    HAL_SANITY_CHECK(NULL != app_mgr.retinaface_hnd);

    app_mgr.mobilefacenet_ops = ALG_GET_OPS(ALG_MODEL_MOBILEFACENET);
    HAL_SANITY_CHECK(NULL != app_mgr.mobilefacenet_ops);

    mobilefacenet_conf.heap_addr = ddr_malloc(mobilefacenet_conf.heap_size);
    HAL_SANITY_CHECK(NULL != mobilefacenet_conf.heap_addr);

    if (HAL_OK != Flash_Read(flashhnd, MOBILEFACENET_MODEL_FLASH_OFF, (uint8_t *)(&aidemo_model_weight), MOBILEFACENET_MODEL_SIZE, 1000)) {
        ts_printf("mobilefacenet_app_task: flash Read weight FAILED!\n");
    }
    mobilefacenet_conf.model = aidemo_model_weight;
    mobilefacenet_conf.model_size = MOBILEFACENET_MODEL_SIZE;

    app_mgr.mobilefacenet_hnd = ALG_INIT(app_mgr.mobilefacenet_ops, &mobilefacenet_conf);
    HAL_SANITY_CHECK(NULL != app_mgr.mobilefacenet_hnd);

    ts_printf("%s init done, CPU_MHZ:%d\n", __func__, CPU_MHZ);

    app_mgr.npu_ready = 1;

    ALG_PERF_INIT(app_mgr.perf, "npu_perf", 50);
    while (1) {
        struct npu_proc_param app_proc_param;

        if (app_mgr.skip_cnt) {
            vTaskDelay(2);
            continue;
        }

        app_mgr.block_cnt = 0;
        if (HAL_QUEU_GET(app_mgr.task_queue, &app_proc_param, 2) != pdPASS) {
            if (!app_mgr.skip_cnt)
                app_mgr.busy = 0;
            continue;
        }

        ALG_PERF_END(app_mgr.perf);
        ALG_PERF_START(app_mgr.perf);

        RetinafacePreArg_t retinaface_pre_param;
        retinaface_pre_param.in = app_proc_param.scale_image;
        retinaface_pre_param.out = retinaface_model_input;
        ret = ALG_CTRL(app_mgr.retinaface_ops, app_mgr.retinaface_hnd, ALG_MODEL_CMD_PRE, &retinaface_pre_param);
        ALG_CACHE_CLEAN(retinaface_model_input, sizeof(retinaface_model_input));

        RetinafaceProcArg_t retinaface_proc_param;
        void *in[1];
        in[0] = retinaface_model_input;
        retinaface_proc_param.in = in;
        retinaface_proc_param.out = NULL;
        ret = ALG_PROC(app_mgr.retinaface_ops, app_mgr.retinaface_hnd, &retinaface_proc_param);
        HAL_SANITY_CHECK(ret == 0);

        RetinafacePostArg_t retinaface_post_param;
        retinaface_post_param.model_output = NULL;
        retinaface_post_param.face_num = 0;
        retinaface_post_param.feature = NULL;
        ret = ALG_CTRL(app_mgr.retinaface_ops, app_mgr.retinaface_hnd, ALG_MODEL_CMD_POST, &retinaface_post_param);
        // ts_printf("%s face_num:%d\n", __func__, retinaface_post_param.face_num);

        if (retinaface_post_param.face_num > NPU_OSD_BOX_MAX)
            retinaface_post_param.face_num = NPU_OSD_BOX_MAX;

        struct npu_osd_info *osd_info = &app_mgr.osd_info[app_mgr.osd_idx];

        HAL_LOCK(osd_info->lock);
        osd_info->valid = 0;

        /* backup the origin feature */
        for (int i = 0; i < retinaface_post_param.face_num; i++) {
            memcpy(&osd_info->feature[i], &retinaface_post_param.feature[i], sizeof(struct retinaface_feature_t));
        }

        RetinafaceScaleArg_t retinaface_scale_param;
        retinaface_scale_param.feature = retinaface_post_param.feature;
        retinaface_scale_param.feature_cnt = retinaface_post_param.face_num;
        retinaface_scale_param.src_h = retinaface_conf.post_conf->det_h;
        retinaface_scale_param.src_w = retinaface_conf.post_conf->det_w;
        retinaface_scale_param.dst_h = retinaface_conf.post_conf->src_h;
        retinaface_scale_param.dst_w = retinaface_conf.post_conf->src_w;
        ALG_CTRL(app_mgr.retinaface_ops, app_mgr.retinaface_hnd, ALG_MODEL_CMD_SCALE,
                &retinaface_scale_param);

        for (int i = 0; i < retinaface_post_param.face_num; i++) {
            struct retinaface_feature_t* feature = &retinaface_post_param.feature[i];
#if 0
            if (i >= 5) {
                memcpy(&osd_info->feature[i], feature, sizeof(struct retinaface_feature_t));
                osd_info->min_val[i] = 10.0f;
                osd_info->min_idx[i] = -1;
                continue;
            }
#endif
            MobilefacenetPreArg_t mobilefacenet_pre_param;
            mobilefacenet_pre_param.in = app_proc_param.raw_image;
            mobilefacenet_pre_param.in_width = app_proc_param.raw_image_width;
            mobilefacenet_pre_param.in_height = app_proc_param.raw_image_height;
            mobilefacenet_pre_param.out = mobilefacenet_model_input;
            mobilefacenet_pre_param.landmark = (float *)&feature->landmark;
            ret = ALG_CTRL(app_mgr.mobilefacenet_ops, app_mgr.mobilefacenet_hnd, ALG_MODEL_CMD_PRE, &mobilefacenet_pre_param);

            ALG_CACHE_CLEAN(mobilefacenet_model_input, sizeof(mobilefacenet_model_input));
            MobilefacenetProcArg_t mobilefacenet_proc_param;
            void *in[1];
            in[0] = mobilefacenet_model_input;
            mobilefacenet_proc_param.in = in;
            mobilefacenet_proc_param.out = NULL;
            ret = ALG_PROC(app_mgr.mobilefacenet_ops, app_mgr.mobilefacenet_hnd, &mobilefacenet_proc_param);
            HAL_SANITY_CHECK(ret == 0);

            MobilefacenetPostArg_t mobilefacenet_post_param;
            mobilefacenet_post_param.vector = NULL;
            mobilefacenet_post_param.min_val = 0.1f;
            mobilefacenet_post_param.min_idx = -1;
            ret = ALG_CTRL(app_mgr.mobilefacenet_ops, app_mgr.mobilefacenet_hnd, ALG_MODEL_CMD_POST, &mobilefacenet_post_param);
            //ts_printf("%s min_val:%f, min_idx:%d\n", __func__,
            //        mobilefacenet_post_param.min_val,
            //        mobilefacenet_post_param.min_idx);

            osd_info->min_val[i] = mobilefacenet_post_param.min_val;
            osd_info->min_idx[i] = mobilefacenet_post_param.min_idx;
        }

        osd_info->det_num = retinaface_post_param.face_num;
        osd_info->valid = 1;
        app_mgr.osd_idx = !app_mgr.osd_idx;
        HAL_UNLOCK(osd_info->lock);
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

int app_npu_face_recognition_init(void)
{
    memset(&app_mgr, 0, sizeof(AppMgr_t));

    alg_init();

    app_npu_ctrl_init();

    app_mgr.rect_mem_pool = alg_mem_pool_init(MAX_RECT_CNT, sizeof(struct rect_label));
    HAL_SANITY_CHECK(NULL != app_mgr.rect_mem_pool);
    INIT_LIST_HEAD(&app_mgr.rect_head);

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

    return 0;
}

extern int _npu_busy;
void app_face_recog_npu_monitor(void)
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
