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
#include "flash.h"
#include <string>
#include <queue>

using namespace cv;

#define GPIO_CTRL_PIN         GPIO_PIN1
#define GPIO_CTRL_PIN_GP      SYS_IP_GPIO3
#define GPIO_CTRL_PIN_PAD     PAD_SDIO0_DATA3
#define GPIO_CTRL_PIN_FUNC    PAD_SDIO0_DATA3_FUNC_GPIO3_1
#define NPU_OSD_BOX_MAX 20
#define PEDESTRAIN_MODEL_FLASH_OFF 0x870000 
#define PEDESTRAIN_MODEL_SIZE 1827388
#define LABEL_IOU_THRESHOLD (0.6f)

struct npu_osd_info {
    SemaphoreHandle_t lock;

    int det_num;
    int valid;
    struct pedestrain_yolov5_bbox_t bbox[NPU_OSD_BOX_MAX];
};

typedef struct {
   struct npu_osd_info osd_info[2];

   struct alg_ops *pedestrain_yolov5_ops;
   void *pedestrain_yolov5_hnd;

   int npu_ready;
   int busy;

   int osd_idx;
   QueueHandle_t task_queue;

   int skip_cnt;
   int block_cnt;

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

class PedestrainQueueSmooth {
private:
    cv::Rect Rect_total;
public:
    bool flag_in;
    std::queue<cv::Rect> queue;
    PedestrainQueueSmooth() {
        flag_in = false;
        Rect_total.x = 0;
        Rect_total.y = 0;
        Rect_total.height = 0;
        Rect_total.width = 0;
    }
    void queueSmoothClear();
    void rectSmooth(cv::Rect& rect);
};

void PedestrainQueueSmooth::queueSmoothClear() {
    while (!queue.empty()) {
        queue.pop();
    }
    flag_in = false;
    Rect_total.x = 0;
    Rect_total.y = 0;
    Rect_total.width = 0;
    Rect_total.height = 0;
}

void PedestrainQueueSmooth::rectSmooth(cv::Rect& rect) {
    if (queue.size() == 0) {
        queue.push(rect);
        Rect_total.x += rect.x;
        Rect_total.y += rect.y;
        Rect_total.width += rect.width;
        Rect_total.height += rect.height;
        flag_in = true;
    } else {
        cv::Rect tmprect = queue.back();
        if (queue.size() < 8) {
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

static AppMgr_t app_mgr;
static PedestrainQueueSmooth queue_smooth[5];

static void reset_all_queue_flag() {
    for (int i = 0; i < 5; i++) {
        queue_smooth[i].flag_in = false;
    }
}

static void process_rect_smooth(cv::Rect& rect) {
    int i = 0;
    float max_iou = 0.0f;
    float tmp_iou = 0.0f;
    bool find_closest_iou = false;
    for (int j = 0; j < 5; j++) {
        if (queue_smooth[j].queue.size() != 0) {
            cv::Rect tmprect = queue_smooth[j].queue.back();
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
        queue_smooth[i].rectSmooth(rect);
    }
}

static void clear_queue_name_not_in() {
    for (int i = 0; i < 5; i++) {
        if (queue_smooth[i].flag_in == false) {
            queue_smooth[i].queueSmoothClear();
        }
    }
}

static uint8_t pedestrain_yolov5_model_input[384 * 228 * 4] __attribute__((aligned(64)));

extern DMA_UNCACHEBLE_DATA uint8_t aidemo_model_weight[2 * 1024 * 1024] __attribute__((aligned(64)));

int app_pedestrain_npu_is_busy(void)
{
    return app_mgr.busy;
}

int app_pedestrain_npu_proc(struct npu_proc_param *param)
{
    if (!app_mgr.npu_ready)
        return 0;

    HAL_QUEU_PUT(app_mgr.task_queue, param, 0);
    app_mgr.busy = 1;
    return 0;
}

#define ATTACK_EN   (1)
int app_pedestrain_npu_osd_handle(void *y, void *uv, int width, int height)
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
#if ATTACK_EN
    cv::Rect detect_rect;
    detect_rect.x = width / 3;
    detect_rect.y = height / 3;
    detect_rect.width = width / 3;
    detect_rect.height = height/ 3;
    nv12_rectangle(y, uv, width, height, detect_rect, cv::Scalar(255, 255, 0), 2);
#endif
    for (int i = 0; i < osd_info->det_num; i++) {
        struct pedestrain_yolov5_bbox_t tmp_bbox;
        struct pedestrain_yolov5_bbox_t *bbox = &tmp_bbox;

        memcpy(&tmp_bbox, &osd_info->bbox[i], sizeof(struct pedestrain_yolov5_bbox_t));

        PedestrainYolov5ScaleArg_t pedestrain_yolov5_scale_param;
        pedestrain_yolov5_scale_param.bbox = bbox;
        pedestrain_yolov5_scale_param.bbox_cnt = 1;
        pedestrain_yolov5_scale_param.src_h = pedestrain_yolov5_conf.post_conf->det_h;
        pedestrain_yolov5_scale_param.src_w = pedestrain_yolov5_conf.post_conf->det_w;
        pedestrain_yolov5_scale_param.dst_h = height;
        pedestrain_yolov5_scale_param.dst_w = width;
        ALG_CTRL(app_mgr.pedestrain_yolov5_ops, app_mgr.pedestrain_yolov5_hnd, ALG_MODEL_CMD_SCALE,
                &pedestrain_yolov5_scale_param);

        cv::Rect rect;
        rect.x = (int)(bbox->center_w - bbox->len_w / 2.0f);
        rect.y = (int)(bbox->center_h - bbox->len_h / 2.0f);
        rect.width = (int)(bbox->len_w);
        rect.height = (int)(bbox->len_h);

        //rect_smooth process
        reset_all_queue_flag();
        process_rect_smooth(rect);
        clear_queue_name_not_in();

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
            point.y = rect.y - 50;
        }
#if ATTACK_EN
        float iou = box_iou(rect, detect_rect);
        if (iou > 0.05f) {
            char label[64];
            snprintf(label, 63, "attack: person %d%%",
                    (int)(100.0f * bbox->cls_confidence + 0.5f));
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(255, 0, 0), 2);
            point.y += 40;
            snprintf(label, 63, "iou %d%%", (int)(100.0f * iou + 0.5f));
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(255, 0, 0), 2);
            nv12_rectangle(y, uv, width, height, rect, cv::Scalar(255, 0, 0), 2);
        } else {
            char label[64];
            snprintf(label, 63, "person %d%%",
                    (int)(100.0f * bbox->cls_confidence + 0.5f));
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(0, 255, 0), 2);
            point.y += 40;
            snprintf(label, 63, "iou %d%%", (int)(100.0f * iou + 0.5f));
            nv12_putText(y, uv, width, height, label,
                    point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(0, 255, 0), 2);
            nv12_rectangle(y, uv, width, height, rect, cv::Scalar(0, 255, 0), 2);
        }
#else
        char label[64];
        snprintf(label, 63, "person %d%%", (int)(100.0f * bbox->cls_confidence + 0.5f));
        nv12_putText(y, uv, width, height, label,
                point, FONT_HERSHEY_SIMPLEX, 1.3f, cv::Scalar(255, 0, 0), 2);

        nv12_rectangle(y, uv, width, height, rect, cv::Scalar(255, 0, 0), 2);
#endif
    }

    HAL_UNLOCK(osd_info->lock);

    return 0;
}

static void app_task(void *param)
{
    int ret;

    npu_init();

    app_mgr.pedestrain_yolov5_ops = ALG_GET_OPS(ALG_MODEL_PEDESTRAIN_YOLOV5);
    HAL_SANITY_CHECK(NULL != app_mgr.pedestrain_yolov5_ops);

    pedestrain_yolov5_conf.heap_addr = ddr_malloc(pedestrain_yolov5_conf.heap_size);
    HAL_SANITY_CHECK(NULL != pedestrain_yolov5_conf.heap_addr);

    FlashHnd_t flashhnd = Flash_Init();
    HAL_SANITY_CHECK(flashhnd != NULL);

    if (HAL_OK != Flash_Read(flashhnd, PEDESTRAIN_MODEL_FLASH_OFF, (uint8_t *)(&aidemo_model_weight), PEDESTRAIN_MODEL_SIZE, 1000)) {
        ts_printf("pedestrain_app_task: flash Read weight FAILED!\n");
    }
    pedestrain_yolov5_conf.model = aidemo_model_weight;
    pedestrain_yolov5_conf.model_size = PEDESTRAIN_MODEL_SIZE;

    app_mgr.pedestrain_yolov5_hnd = ALG_INIT(app_mgr.pedestrain_yolov5_ops, &pedestrain_yolov5_conf);
    HAL_SANITY_CHECK(NULL != app_mgr.pedestrain_yolov5_hnd);

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

        PedestrainYolov5PreArg_t pedestrain_yolov5_pre_param;
        pedestrain_yolov5_pre_param.in = app_proc_param.scale_image;
        pedestrain_yolov5_pre_param.out = pedestrain_yolov5_model_input;
        ret = ALG_CTRL(app_mgr.pedestrain_yolov5_ops, app_mgr.pedestrain_yolov5_hnd, ALG_MODEL_CMD_PRE, &pedestrain_yolov5_pre_param);
        sys_dcache_clean_by_addr(pedestrain_yolov5_model_input, sizeof(pedestrain_yolov5_model_input));

        PedestrainYolov5ProcArg_t pedestrain_yolov5_proc_param;
        void *in[1];
        in[0] = pedestrain_yolov5_model_input;
        pedestrain_yolov5_proc_param.in = in;
        pedestrain_yolov5_proc_param.out = NULL;
        ret = ALG_PROC(app_mgr.pedestrain_yolov5_ops, app_mgr.pedestrain_yolov5_hnd, &pedestrain_yolov5_proc_param);
        HAL_SANITY_CHECK(ret == 0);

        PedestrainYolov5PostArg_t pedestrain_yolov5_post_param;
        pedestrain_yolov5_post_param.model_output = NULL;
        pedestrain_yolov5_post_param.det_num = 0;
        pedestrain_yolov5_post_param.bbox = NULL;
        ret = ALG_CTRL(app_mgr.pedestrain_yolov5_ops, app_mgr.pedestrain_yolov5_hnd, ALG_MODEL_CMD_POST, &pedestrain_yolov5_post_param);
        //ts_printf("%s det_num:%d\n", __func__, pedestrain_yolov5_post_param.det_num);

        struct npu_osd_info *osd_info = &app_mgr.osd_info[app_mgr.osd_idx];
		HAL_LOCK(osd_info->lock);
        osd_info->valid = 0;

        if (pedestrain_yolov5_post_param.det_num > NPU_OSD_BOX_MAX)
            pedestrain_yolov5_post_param.det_num = NPU_OSD_BOX_MAX;

        osd_info->det_num = pedestrain_yolov5_post_param.det_num;
        memcpy(&osd_info->bbox[0], pedestrain_yolov5_post_param.bbox,
                pedestrain_yolov5_post_param.det_num * sizeof(struct pedestrain_yolov5_bbox_t));

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

int app_npu_pedestrain_init(void)
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

    return 0;
}

extern int _npu_busy;
void app_pedestrain_npu_monitor(void)
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
