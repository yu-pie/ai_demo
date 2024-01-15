#include <iostream>
#include <vector>
#include "alg_module.h"
// #include "npu_init.h"
#include "alg_conf.h"
#include "test_gesture_detection.h"
// #include "app_npu.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
#include "gesture_detection_post.h"
#include "gesture_keypoint_detection_post.h"
#include "alg_perf.h"

#include "flash.h"


#pragma GCC push_options
#pragma GCC optimize ("O3")

// #include <fstream> 
using namespace std;

const float eps = 3;

static void my_assert(bool flag, int idx){
    if (!flag) {
        std::cout << "@@@@error, index = " <<  idx << std::endl; 
    }
}

typedef struct {
    // struct npu_osd_info osd_info[2];

    struct alg_ops *gesture_detection_ops;
    // struct alg_ops *gesture_keypoint_ops;

    void *gesture_detection_hnd;
    // void *gesture_keypoint_hnd;

    int npu_ready;
    int busy;

    int osd_idx;
    // QueueHandle_t task_queue;

    int skip_cnt;
    int block_cnt;

    // GPIOHnd_t gpioHnd;

    ALG_PERF_DEF(perf1);
} AppMgr_t1;

typedef struct {
    // struct npu_osd_info osd_info[2];

    // struct alg_ops *gesture_detection_ops;
    struct alg_ops *gesture_keypoint_ops;

    // void *gesture_detection_hnd;
    void *gesture_keypoint_hnd;

    int npu_ready;
    int busy;

    int osd_idx;
    // QueueHandle_t task_queue;

    int skip_cnt;
    int block_cnt;

    // GPIOHnd_t gpioHnd;

    ALG_PERF_DEF(perf2);
} AppMgr_t2;

static AppMgr_t1 app_mgr1;
static AppMgr_t2 app_mgr2;

#if 1
/* NPU side*/
static uint8_t _output0[] = {
#include "./data/hand/output0_transpose.txt"
};
static uint8_t _output1[] = {
#include "./data/hand/output1_transpose.txt"
};
static uint8_t _output2[] = {
#include "./data/hand/output2_transpose.txt"
};
static uint8_t _output3[] = {
#include "./data/hand/output3_transpose.txt"
};
static uint8_t _bbox[] = {
#include "./data/hand/det_bboxes_last.txt"
};
static uint8_t _score[] = {
#include "./data/hand/det_scores_last.txt"
};

int test_hand_dec(){

    app_mgr1.gesture_detection_ops = ALG_GET_OPS(ALG_MODEL_GESTURE_DETECTION);
    HAL_SANITY_CHECK(NULL != app_mgr1.gesture_detection_ops);

    app_mgr1.gesture_detection_hnd = ALG_INIT(app_mgr1.gesture_detection_ops, &gesture_detection_conf);
    ALG_PERF_INIT(app_mgr1.perf1, "npu_perf", 50);

    std::vector<float *> outputs;
    std::vector<float> output0(1600 * 33), output1(400 * 33), output2(100 * 33), output3(25 * 33);
    for (int i = 0; i < (1600 * 33); i++) {
        output0[i] = *((float *)_output0 + i);
    }
    for (int i = 0; i < (400 * 33); i++) {
        output1[i] = *((float *)_output1 + i);
    }
    for (int i = 0; i < (100 * 33); i++) {
        output2[i] = *((float *)_output2 + i);
    }
    for (int i = 0; i < (25 * 33); i++) {
        output3[i] = *((float *)_output3 + i);
    }

    outputs.push_back(output0.data());
    outputs.push_back(output1.data());
    outputs.push_back(output2.data());
    outputs.push_back(output3.data());

    GestureDetectionPostArg_t gesture_detection_post_param;

    // 定义并分配内存给tmp_output  
    void **tmp_output = new void*[outputs.size()];  
    for (int i = 0; i < outputs.size(); ++i) {  
        tmp_output[i] = static_cast<void*>(outputs[i]); // 将float*转换为void*  
    }

    gesture_detection_post_param.model_output = tmp_output;
    gesture_detection_post_param.det_num = 0;
    gesture_detection_post_param.bbox = NULL;
    ALG_PERF_START(app_mgr1.perf1);
    uint32_t start = get_core_perf_cnt();
    int ret = ALG_CTRL(app_mgr1.gesture_detection_ops, app_mgr1.gesture_detection_hnd, ALG_MODEL_CMD_POST, &gesture_detection_post_param);
    uint32_t end = get_core_perf_cnt(); 
    uint32_t used = end - start;
    std::cout << "hand detection post_proc time cost: " << int(used / CPU_MHZ) << "us."<< std::endl;
    ALG_PERF_END(app_mgr1.perf1);

    std::vector<float> det_bboxes_last_npy(4);
    std::vector<float> det_scores_last_npy(1);
    // std::cout << "@@sizeof bbox" << sizeof(_bbox) << std::endl;
    // std::cout << "@@sizeof score" << sizeof(_score) << std::endl;
    for (int i = 0; i < 4; i++) {
        det_bboxes_last_npy[i] = *((float *)_bbox + i);
    }
    for (int i = 0; i < 1; i++) {
        det_scores_last_npy[i] = *((float *)_score + i);
    }

    // for (int i = 0; i < gesture_detection_post_param.det_num; i++)
    // {
        std::cout << "refer.x1 = " << gesture_detection_post_param.bbox[0].box.x1 << " board_proc_result.x1 = " << det_bboxes_last_npy[0] << "#" << std::endl;
        std::cout << "refer.y1 = " << gesture_detection_post_param.bbox[0].box.y1 << " board_proc_result.y1 = " << det_bboxes_last_npy[1] << "#" << std::endl;
        std::cout << "refer.x2 = " << gesture_detection_post_param.bbox[0].box.x2 << " board_proc_result.x2 = " << det_bboxes_last_npy[2] << "#" << std::endl;
        std::cout << "refer.y2 = " << gesture_detection_post_param.bbox[0].box.y2 << " board_proc_result.y2 = " << det_bboxes_last_npy[3] << "#" << std::endl;
        std::cout << "refer.score = " << gesture_detection_post_param.bbox[0].score  << " board_proc_result.score = " << det_scores_last_npy[0] << "#" << std::endl;
        // my_assert((abs(gesture_detection_post_param.bbox[i].box.x1 - det_bboxes_last_npy[i * 4 + 0]) < 1), i);
        // my_assert((abs(gesture_detection_post_param.bbox[i].box.y1 - det_bboxes_last_npy[i * 4 + 1]) < 1), i);
        // my_assert((abs(gesture_detection_post_param.bbox[i].box.x2 - det_bboxes_last_npy[i * 4 + 2]) < 1), i);
        // my_assert((abs(gesture_detection_post_param.bbox[i].box.y2 - det_bboxes_last_npy[i * 4 + 3]) < 1), i);
        // my_assert((abs(gesture_detection_post_param.bbox[i].score - det_scores_last_npy[i]) < eps), i);
    // }

    // std::cout << "test interface passed!" << std::endl;

    delete[] tmp_output;

    return 0;

}
#endif

#if 0
static uint8_t _output0[] = {
#include "./data/hand/predicts.txt"
};

static uint8_t _results[] = {
#include "./data/hand/results.txt"
};

int test_keypoint_dec(){

    std::cout << " !!!!!@@@@@" << std::endl;
    app_mgr2.gesture_keypoint_ops = ALG_GET_OPS(ALG_MODEL_GESTURE_KEYPOINT_DETECTION);
    HAL_SANITY_CHECK(NULL != app_mgr2.gesture_keypoint_ops);

    app_mgr2.gesture_keypoint_hnd = ALG_INIT(app_mgr2.gesture_keypoint_ops, &gesture_keypoint_detection_conf);
    ALG_PERF_INIT(app_mgr2.perf2, "npu_perf", 50);

    std::vector<float> output0(21 * 48 * 48);
    for (int i = 0; i < (21 * 48 *48); i++) {
        output0[i] = *((float *)_output0 + i);
    }

    float* rawArray = output0.data();  
  
    // 声明一个void*类型的指针  
    void* voidPtr = static_cast<void*>(rawArray);  

    GestureKeypointDetectionPostArg_t gesture_keypoint_post_param;

    gesture_keypoint_post_param.model_output = voidPtr;
    // gesture_keypoint_post_param.output = NULL;
    ALG_PERF_START(app_mgr2.perf2);
    uint32_t start = get_core_perf_cnt();
    int ret = ALG_CTRL(app_mgr2.gesture_keypoint_ops, app_mgr2.gesture_keypoint_hnd, ALG_MODEL_CMD_POST, &gesture_keypoint_post_param);
    uint32_t end = get_core_perf_cnt(); 
    uint32_t used = end - start;
    std::cout << "time cost: " << int(used / CPU_MHZ) << "us."<< std::endl;
    ALG_PERF_END(app_mgr2.perf2);

    std::vector<float> last_result_npy(21 * 3);
    // std::cout << "@@sizeof bbox" << sizeof(_bbox) << std::endl;
    // std::cout << "@@sizeof score" << sizeof(_score) << std::endl;
    for (int i = 0; i < (21 * 3); i++) {
        last_result_npy[i] = *((float *)_results + i);
    }

    for (int i = 0; i < 21; i++)
    {
        
        // std::cout << "## \t" << "##" << last_result_npy[3 * i] << "#" << std::endl;
        // std::cout << "## \t" << "##" << last_result_npy[3 * i + 1] << "#" << std::endl;
        // std::cout << "## \t" << "##" << last_result_npy[3 * i + 2] << "#" << std::endl;

        std::cout << "## \t" << gesture_keypoint_post_param.output.key_point[i].x << "##" << last_result_npy[3 * i] << "#" << std::endl;
        std::cout << "## \t" << gesture_keypoint_post_param.output.key_point[i].y << "##" << last_result_npy[3 * i + 1] << "#" << std::endl;
        std::cout << "## \t" << gesture_keypoint_post_param.output.key_point[i].score << "##" << last_result_npy[3 * i + 2] << "#" << std::endl;
    }

    // std::cout << "test interface passed!" << std::endl;

    return 0;
}
#endif

#if 0
uint8_t warp_in[] = {
#include "./data/hand/img_warp.txt"
};
uint8_t warp_out[192 * 192 * 3];
static uint8_t warp_result[] = {
#include "./data/hand/warp_result.txt"
};

int test_warp(){
    // cv::Mat img = cv::imread("/data/hand/like_11.jpg");
    // uint8_t *_img = (uint8_t *)img.data;
    std::cout << "test_warp()" << "sizeof warp_in =" << sizeof(warp_in) << std::endl;
    // BBox _bbox = {1089.0f, 263.0f, 1268.0f, 514.0f};
    BBox _bbox = {1127.84f, 283.75f, 1229.64f, 495.22f};
    cv::Mat img_warp;

    app_mgr2.gesture_keypoint_ops = ALG_GET_OPS(ALG_MODEL_GESTURE_KEYPOINT_DETECTION);
    // HAL_SANITY_CHECK(NULL != app_mgr2.gesture_keypoint_ops);

    app_mgr2.gesture_keypoint_hnd = ALG_INIT(app_mgr2.gesture_keypoint_ops, &gesture_keypoint_detection_conf);
    // ALG_PERF_INIT(app_mgr2.perf2, "npu_perf", 50);


    // std::vector<uint8_t> output0(1920 * 1080 * 3);
    // for (int i = 0; i < (1920 * 1080 * 3); i++) {
    //     output0[i] = *((uint8_t *)warp_in + i);
    // }

    // uint8_t* rawArray = output0.data();  
  
    // // 声明一个void*类型的指针  
    // void* voidPtr = static_cast<void*>(rawArray);  


    // uint8_t *tmp_warp_in;
    // tmp_warp_in = (uint8_t *)malloc(sizeof(warp_in));
    // for (uint32_t i = 0; i < 6220800; i++) {
    //     tmp_warp_in[i] = warp_in[i];
    // }
    GestureKeypointDetectionPreArg_t gesture_keypoint_pre_param;
    gesture_keypoint_pre_param.in = warp_in;
    // gesture_keypoint_pre_param.in = tmp_warp_in;
    // gesture_keypoint_pre_param.in = voidPtr;
    memset(warp_out, 0, 192 * 192 * 3);
    gesture_keypoint_pre_param.out = warp_out;
    gesture_keypoint_pre_param.pre_conf.bbox.box = _bbox;
    gesture_keypoint_pre_param.pre_conf.img_width = gesture_keypoint_detection_pre_conf.img_width;
    gesture_keypoint_pre_param.pre_conf.img_height = gesture_keypoint_detection_pre_conf.img_height;
    gesture_keypoint_pre_param.pre_conf.img_channel = gesture_keypoint_detection_pre_conf.img_channel;
    gesture_keypoint_pre_param.pre_conf.input_width = gesture_keypoint_detection_pre_conf.input_width; 
    gesture_keypoint_pre_param.pre_conf.input_height = gesture_keypoint_detection_pre_conf.input_height; 
    gesture_keypoint_pre_param.pre_conf.expanding_ratio = gesture_keypoint_detection_pre_conf.expanding_ratio; 
    gesture_keypoint_pre_param.pre_conf.aspect_ratio = gesture_keypoint_detection_pre_conf.aspect_ratio; 
    gesture_keypoint_pre_param.pre_conf.pixel_std = gesture_keypoint_detection_pre_conf.pixel_std;
    gesture_keypoint_pre_param.pre_conf.scale_rate = gesture_keypoint_detection_pre_conf.scale_rate;
    memcpy(&gesture_keypoint_pre_param.pre_conf.img_norm, &gesture_keypoint_detection_pre_conf.img_norm, sizeof(ImageNorm_t));

    std::cout << "@@@test_warp() ctrl" << std::endl;
    std::cout << "@@@" << warp_result[0] << "@@" << std::endl;
    // ALG_PERF_START(app_mgr2.perf2);
    uint32_t start = get_core_perf_cnt();
    int ret = ALG_CTRL(app_mgr2.gesture_keypoint_ops, app_mgr2.gesture_keypoint_hnd, ALG_MODEL_CMD_PRE, &gesture_keypoint_pre_param);
    uint32_t end = get_core_perf_cnt(); 
    uint32_t used = end - start;
    std::cout << "time cost: " << int(used / CPU_MHZ) << "us."<< std::endl;
    // ALG_PERF_END(app_mgr2.perf2);
    std::cout << "@@@after test_warp() ctrl" <<std::endl;

    // std::vector<uint8_t> img_warp_gt(192 * 192 * 3);
    // for (int i = 0; i < (192 * 192 * 3); i++) {
    //     img_warp_gt[i] = *((uint8_t *)warp_result + i);
    // }

// #define my_max(x, y) (x > y ? x : y)
//     for (int i = 0; i < 192; i++)
//     {
//         for (int j = 0; j < 192; j++)
//         {
//             // for (int k = 0; k < 3; k++)
//             // {
//             //     // uint8_t p = img_warp.at<cv::Vec3b>(i, j)[k];
//             //     // uint8_t tmp_p = *((uint8_t *)gesture_keypoint_pre_param.out + i * 192 * 3 + j * 3 + k);
//             //     // uint8_t p_ = img_warp_gt[i * 192 * 3 + j * 3 + k];
//             //     float diff = abs(float(tmp_p) - p_);
//             //     // assert (diff < eps);
//             //     if (diff >= eps) {
//             //         std::cout << "error, diff >= eps!" <<std::endl;
//             //         std::cout << "idx = " << (i * 192 * 3 + j * 3 + k) << " result = " << float(tmp_p) << " refer = " << p_ << std::endl;
//             //         return 1;
//             //     }
//             // }
//             int diff1 = 0;
//             int diff2 = 0;
//             int diff3 = 0;
//             int tmp = i * 192 + j;
//             diff1 = abs((int)warp_out[3 * tmp] - (int)warp_result[3 * tmp + 2]);
//             diff2 = abs((int)warp_out[3 * tmp + 1] - (int)warp_result[3 * tmp + 1]);
//             diff2 = abs((int)warp_out[3 * tmp + 2] - (int)warp_result[3 * tmp + 0]);
//             int diff = my_max(my_max(diff1, diff2), diff3);
//             if (diff > eps) {
//                 std::cout << "error, diff >= eps! diff = " << diff <<std::endl;
//                 std::cout << (int)warp_out[3 * tmp] << (int)warp_out[3 * tmp + 1] << (int)warp_out[3 * tmp + 2] << std::endl;  
//                 std::cout << (int)warp_result[3 * tmp] << (int)warp_result[3 * tmp + 1] << (int)warp_result[3 * tmp + 2] << std::endl;  
//                 // std::cout << "idx = " << i << " result = " << (int)warp_out[i] << " refer = " << (int)warp_result[i] << std::endl;
//                 return 1;
//             }
//         }
//     }
//     std::cout << "test succeed." <<std::endl;

    // for (int i = 0; i < 192 * 192; i++)
    // {
    //     uint8_t tmp_p = *((uint8_t *)gesture_keypoint_pre_param.out + i);
    //     // uint8_t p_ = img_warp_gt[i];
    //     std::cout << "idx = " << (3 * i) << " result = " << (int)warp_out[3 * i] << " refer = " << (int)warp_result[3 * i + 2] << std::endl;
    //     std::cout << "idx = " << (3 * i + 1) << " result = " << (int)warp_out[3 * i + 1] << " refer = " << (int)warp_result[3 * i + 1] << std::endl;
    //     std::cout << "idx = " << (3 * i + 2) << " result = " << (int)warp_out[3 * i + 2] << " refer = " << (int)warp_result[3 * i] << std::endl;
    //     // std::cout << "idx = " << i << " result = " << (int)warp_out[i] << std::endl;
    // }
    return 0;
}
#endif


#if 0
static float _keypoints[] = {
// #include "./data/hand/test_feat.txt"
#include "./data/hand/output_4_cls_label_v2_new.txt"
};
static uint8_t _gestures[] = {
#include "./data/hand/test_lab.txt"
};
// std::vector<float> keypoints(42);
// for (int i = 0; i < 42; i++) {
//     keypoints[i] = *((float *)_keypoints + i);
// }
void test_classfy() 
{
    gesture_type_t type;
    std::cout << "@@@@@@!" <<std::endl;
    gesture_keypoint_detection_keypoint_t kp_points[21];
    for (int i = 0; i < 1000; i++) {
        for(int j = 0; j < 21; j ++) {
            kp_points[j].x = *((float *)_keypoints + 2 * j + 4 + 4 * i + 42 * i + i + i);
            kp_points[j].y = *((float *)_keypoints + 2 * j + 1 + 4 + 4 * i + 42 * i + i + i);
        }
        
        // ALG_PERF_INIT(app_mgr2.perf2, "GESTURE_JUDGE npu_perf", 1);
        // ALG_PERF_START(app_mgr2.perf2);
        // uint32_t start = get_core_perf_cnt();
        type = gesture_judge(kp_points);
        // uint32_t end = get_core_perf_cnt(); 
        // uint32_t used = end - start;
        // std::cout << "time cost: " << int(used / CPU_MHZ) << "us."<< std::endl;
        // ALG_PERF_END(app_mgr2.perf2);
        // if (type == _gestures[0]) {
        if (type == (int)*((float *)_keypoints + 4 + 42  + 1 + 4 * i + 42 * i + i + i)) {
            std::cout << "succeed, type = " << type << std::endl;
        } else {
            std::cout << "fail, idx = " << i << std::endl << "type = " << (int)type << "\t" << "refer = " << (int)*((float *)_keypoints + 4 + 42 + 4 * i + 42 * i + i) << std::endl;
            // std::cout << "refer = " << (int)_gestures[0]  << "\t##" << (int)_gestures[1]  << "\t##" << (int)_gestures[2]  << "\t##" << (int)_gestures[3]  << "\t" << std::endl;
        }
    }
    // gesture_type_t type;
    // // ALG_PERF_INIT(app_mgr2.perf2, "GESTURE_JUDGE npu_perf", 1);
    // // ALG_PERF_START(app_mgr2.perf2);
    // uint32_t start = get_core_perf_cnt();
    // type = gesture_judge(kp_points);
    // uint32_t end = get_core_perf_cnt(); 
    // uint32_t used = end - start;
    // std::cout << "time cost: " << int(used / CPU_MHZ) << "us."<< std::endl;
    // // ALG_PERF_END(app_mgr2.perf2);
    // if (type == _gestures[0]) {
    //     std::cout << "succeed" << std::endl;
    // } else {
    //     std::cout << "fail" << std::endl << "type = " << (int)type << "\t" << "refer = " << (int)_gestures[0] << std::endl;
    //     // std::cout << "refer = " << (int)_gestures[0]  << "\t##" << (int)_gestures[1]  << "\t##" << (int)_gestures[2]  << "\t##" << (int)_gestures[3]  << "\t" << std::endl;
    // }
}

#endif

#if 1
#define GESTURE_DETECTION_MODEL_FLASH_OFF 0x6b0000
#define GESTURE_DETECTION_MODEL_SIZE
#define GESTURE_KEYPOINT_MODEL_FLASH_OFF 0x6b0000
#define GESTURE_KEYPOINT_MODEL_SIZE 2355788
DMA_UNCACHEBLE_DATA uint8_t aidemo_detection_model_weight[3 * 1024 * 1024] __attribute__((aligned(64)));
uint8_t model_input[] __attribute__((aligned(64))) = {
#include "./data/hand/indata.txt"
};
static uint8_t gesture_hand_keypoint_detection_model[] __attribute__((aligned(64))) = {
    #include "./data/hand/gesture_detection.txt"
};
// uint8_t model_output0[40 * 40 * 33 * 4];
// uint8_t model_output1[400 * 33 * 4];
// uint8_t model_output2[100 * 33 * 4];
// uint8_t model_output3[25 * 33 * 4];
uint8_t model_output[48 * 48 * 21 * 2 * 4];
void test_model(){

    std::cout << " !!!!!@@@@@" << std::endl;
    // app_mgr1.gesture_detection_ops = ALG_GET_OPS(ALG_MODEL_GESTURE_DETECTION);
    // HAL_SANITY_CHECK(NULL != app_mgr1.gesture_detection_ops);
    app_mgr2.gesture_keypoint_ops = ALG_GET_OPS(ALG_MODEL_GESTURE_KEYPOINT_DETECTION);
    HAL_SANITY_CHECK(NULL != app_mgr2.gesture_keypoint_ops);

    // gesture_detection_conf.heap_addr = ddr_malloc(gesture_detection_conf.heap_size);
    // HAL_SANITY_CHECK(NULL != gesture_detection_conf.heap_addr);
    gesture_keypoint_detection_conf.heap_addr = ddr_malloc(gesture_keypoint_detection_conf.heap_size);
    HAL_SANITY_CHECK(NULL != gesture_keypoint_detection_conf.heap_addr);

    FlashHnd_t flashhnd = Flash_Init();
    HAL_SANITY_CHECK(flashhnd != NULL);

    //TODO config aidemo_detection_model_weight
    // if (HAL_OK != Flash_Read(flashhnd, GESTURE_DETECTION_MODEL_FLASH_OFF, (uint8_t *)(&aidemo_detection_model_weight), GESTURE_DETECTION_MODEL_SIZE, 1000)) {
    //     ts_printf("gesture_app_task: flash Read gesture detec weight FAILED!\n");
    // }
    // gesture_detection_conf.model = aidemo_detection_model_weight;
    // gesture_detection_conf.model_size = GESTURE_DETECTION_MODEL_SIZE;

    // if (HAL_OK != Flash_Read(flashhnd, GESTURE_KEYPOINT_MODEL_FLASH_OFF, (uint8_t *)(&aidemo_detection_model_weight), GESTURE_KEYPOINT_MODEL_SIZE, 1000)) {
    //     ts_printf("gesture_app_task: flash Read gesture keypoint detec weight FAILED!\n");
    // }
    // gesture_keypoint_detection_conf.model = aidemo_detection_model_weight;
    // gesture_keypoint_detection_conf.model_size = GESTURE_KEYPOINT_MODEL_SIZE;

    gesture_keypoint_detection_conf.model = gesture_hand_keypoint_detection_model;
    gesture_keypoint_detection_conf.model_size = sizeof(gesture_hand_keypoint_detection_model);

    // app_mgr1.gesture_detection_hnd = ALG_INIT(app_mgr1.gesture_detection_ops, &gesture_detection_conf);
    // HAL_SANITY_CHECK(NULL != app_mgr1.gesture_detection_hnd);

    app_mgr2.gesture_keypoint_hnd = ALG_INIT(app_mgr2.gesture_keypoint_ops, &gesture_keypoint_detection_conf);
    HAL_SANITY_CHECK(NULL != app_mgr2.gesture_keypoint_hnd);

    // GestureDetectionProcArg_t gesture_detection_proc_param;
    // // void *in[1];
    // // in[0] = gesture_detec_model_input;
    // gesture_detection_proc_param.in = model_input;
    // gesture_detection_proc_param.out = NULL;
    // uint32_t start = get_core_perf_cnt();
    // int ret = ALG_PROC(app_mgr1.gesture_detection_ops, app_mgr1.gesture_detection_hnd, &gesture_detection_proc_param);
    // uint32_t end = get_core_perf_cnt(); 
    // uint32_t used = end - start;

    // void **tmp_output = new void*[4];  
    // tmp_output[0] = static_cast<void*>(model_output0); // 将float*转换为void*  
    // tmp_output[1] = static_cast<void*>(model_output1);
    // tmp_output[2] = static_cast<void*>(model_output2);
    // tmp_output[3] = static_cast<void*>(model_output3);

    // memset(model_output0, 0, 1600 * 3 * 4);
    // memset(model_output1, 0, 400 * 3 * 4);
    // memset(model_output2, 0, 100 * 3 * 4);
    // memset(model_output3, 0, 25 * 3 * 4);
    GestureKeypointDetectionProcArg_t gesture_keypoint_detection_proc_param;
    void *in[1];
    in[0] = model_input;
    // ALG_CACHE_CLEAN(in[0], 64);
    gesture_keypoint_detection_proc_param.in = in;
    // gesture_keypoint_detection_proc_param.out = (void **)(&model_output);
    gesture_keypoint_detection_proc_param.out = NULL;
    uint32_t start = get_core_perf_cnt();
    int ret = ALG_PROC(app_mgr2.gesture_keypoint_ops, app_mgr2.gesture_keypoint_hnd, &gesture_keypoint_detection_proc_param);
    uint32_t end = get_core_perf_cnt(); 
    uint32_t used = end - start;
    std::cout << "time cost: " << int(used / CPU_MHZ) << "us."<< std::endl;

    // delete []tmp_output;
}

#endif

#pragma GCC pop_options
