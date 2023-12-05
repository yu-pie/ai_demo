#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "gesture_yolov5_post.h"
#include "ts_print.h"

#pragma GCC push_options
#pragma GCC optimize ("O3")

#define TEST_PRINTF         //ts_printf

static inline float _sigmoid(float val)
{
    return 1.0f / (1.0f + expf(-val));
}

static inline float _iou(struct gesture_yolov5_bbox_t* bbox1, struct gesture_yolov5_bbox_t* bbox2) {
    float box1_lh = bbox1->center_h - bbox1->len_h / 2;
    float box2_lh = bbox2->center_h - bbox2->len_h / 2;
    float y1_max = box1_lh > box2_lh ? box1_lh : box2_lh;
    float box1_lw = bbox1->center_w - bbox1->len_w / 2;
    float box2_lw = bbox2->center_w - bbox2->len_w / 2;
	float x1_max = box1_lw > box2_lw ? box1_lw : box2_lw;
    float box1_rh = bbox1->center_h + bbox1->len_h / 2;
    float box2_rh = bbox2->center_h + bbox2->len_h / 2;
	float y2_min = box1_rh < box2_rh ? box1_rh : box2_rh;
    float box1_rw = bbox1->center_w + bbox1->len_w / 2;
    float box2_rw = bbox2->center_w + bbox2->len_w / 2;
	float x2_min = box1_rw < box2_rw ? box1_rw : box2_rw;
    float x_diff = x2_min - x1_max;
    x_diff = x_diff > 0.0f ? x_diff : 0.0f;
    float y_diff = y2_min - y1_max;
    y_diff = y_diff > 0.0f ? y_diff : 0.0f;
	float intersection_area = x_diff * y_diff;
    float union_area = bbox1->len_w * bbox1->len_h + bbox2->len_w * bbox2->len_h - intersection_area;
	float ratio = intersection_area / union_area;
    return ratio;
}

static int _confidence_compare(const void* e1, const void* e2)
{
    float diff = ((struct gesture_yolov5_bbox_t*)e1)->confidence - 
        ((struct gesture_yolov5_bbox_t*)e2)->confidence;
    if (diff < 0.0f) {
        return 1;
    } else if (diff > 0.0f) {
        return -1;
    } else {
        return 0;
    }
}

static inline int _nms(struct gesture_yolov5_bbox_t* bbox, int len, float iou_threshold)
{
    /* sort score as descending order */
    qsort(bbox, len, sizeof(struct gesture_yolov5_bbox_t), _confidence_compare);
    /* nms */
    int new_idx = 0;
    for (int i = 0; i < len; i++) {
        /* it means it's already deleted from the list if it's confidence is 0*/
        if (bbox[i].confidence < 0.0f) {
            continue;
        }
        /* copy the selected bbox if it's not deleted */
        memcpy(&bbox[new_idx], &bbox[i], sizeof(struct gesture_yolov5_bbox_t));
        /* delete ovelapped bbox */
        for (int j = i + 1; j < len; j++) {
            if (bbox[j].confidence > 0.0f) {
                float iou = _iou(&(bbox[new_idx]), &(bbox[j]));
                // TEST_PRINTF("iou between %d and %d is %f\n", new_idx, j, iou);
                if (iou > iou_threshold) {
                    bbox[j].confidence = -1.0f;
                }
            }
        }
        new_idx++;
    }
    len = new_idx;
    return len;
}

#define _CLIP(v, lo, hi)    ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

int gesture_yolov5_scale_coordinate(struct gesture_yolov5_bbox_t* bbox, int num, int det_h, int det_w, int src_h, int src_w)
{
    float gain_h = ((float)det_h) / ((float)src_h);
    float gain_w = ((float)det_w) / ((float)src_w);
    float gain = gain_h < gain_w ? gain_h : gain_w;
    float pad_h = (((float)det_h) - ((float)src_h) * gain) / 2.0f;
    float pad_w = (((float)det_w) - ((float)src_w) * gain) / 2.0f;
    for (int i = 0; i < num; i++) {
        bbox[i].center_h = (bbox[i].center_h - pad_h) / gain;
        bbox[i].center_w = (bbox[i].center_w - pad_w) / gain;
        bbox[i].len_h = bbox[i].len_h / gain;
        bbox[i].len_w = bbox[i].len_w / gain;
        float h0, h1, w0, w1;
        h0 = bbox[i].center_h - bbox[i].len_h / 2.0f;
        w0 = bbox[i].center_w - bbox[i].len_w / 2.0f;
        h1 = bbox[i].center_h + bbox[i].len_h / 2.0f;
        w1 = bbox[i].center_w + bbox[i].len_w / 2.0f;
        h0 = _CLIP(h0, 0, src_h - 1);
        w0 = _CLIP(w0, 0, src_w - 1);
        h1 = _CLIP(h1, 0, src_h - 1);
        w1 = _CLIP(w1, 0, src_w - 1);
        bbox[i].len_h = h1 - h0;
        bbox[i].len_w = w1 - w0;
        bbox[i].center_h = (h1 + h0) / 2.0f;
        bbox[i].center_w = (w1 + w0) / 2.0f;
    }
    return num;
}

gesture_yolov5_post_hnd_t gesture_yolov5_post_init(struct gesture_yolov5_post_conf_t* conf)
{
    struct gesture_yolov5_post_mgr_t* mgr = (struct gesture_yolov5_post_mgr_t*)malloc(sizeof(struct gesture_yolov5_post_mgr_t));
    memset(mgr, 0, sizeof(struct gesture_yolov5_post_mgr_t));
    memcpy(&mgr->conf, conf, sizeof(struct gesture_yolov5_post_conf_t));
    mgr->confidence_threshold_log = -logf((1.0f - conf->confidence_threshold) / conf->confidence_threshold);
    TEST_PRINTF("confidence_threshold_log = %f\n", mgr->confidence_threshold_log);
    if (mgr->conf.max_det < 128) {
        mgr->conf.max_det = 128;
    }
    mgr->bbox = (void*)malloc(mgr->conf.max_det * sizeof(struct gesture_yolov5_bbox_t));
    memset(mgr->bbox, 0, mgr->conf.max_det * sizeof(struct gesture_yolov5_bbox_t));

    return (gesture_yolov5_post_hnd_t)mgr;
}

int gesture_yolov5_post_proc(gesture_yolov5_post_hnd_t hnd, void** model_out)
{
    struct gesture_yolov5_bbox_t* bbox = (struct gesture_yolov5_bbox_t*)hnd->bbox;
    int box_cnt = 0;
    /* filter the boxes */
    for (int i = 0; i < 3; i++) {
        struct gesture_yolov5_feature_t* feature = (struct gesture_yolov5_feature_t*)model_out[i];
        int nh = hnd->conf.feature_shapes[i][0];
        int nw = hnd->conf.feature_shapes[i][1];
        int nc = hnd->conf.feature_shapes[i][2];
        for (int hh = 0; hh < nh; hh++) {
            for (int ww = 0; ww < nw; ww++) {
                for (int cc = 0; cc < nc / (sizeof(struct gesture_yolov5_feature_t) / sizeof(float)); cc++) {
                    /* filter by obj_confidence */
                    if (feature->obj_confidence < hnd->confidence_threshold_log) {
                        feature++;
                        continue;
                    }
                    float max_cls_confidence = -1e6;
                    int label_idx = -1;
                    for (int i = 0; i < GESTURE_TYPE_NUM; i++) {
                        if (max_cls_confidence < feature->cls_confidence[i]) {
                            max_cls_confidence = feature->cls_confidence[i];
                            label_idx = i;
                        }
                    }
#if 0
                    /* filter by max_cls_confidence */
                    if (max_cls_confidence < hnd->confidence_threshold_log) {
                        feature++;
                        continue;
                    }
#endif
                    /* calculate final confidence, store it in cls_confidence and filter by final confidence */
                    // bbox->confidence = _sigmoid(feature->obj_confidence);
                    bbox->confidence = _sigmoid(max_cls_confidence);
                    if (bbox->confidence < hnd->conf.confidence_threshold) {
                        feature++;
                        continue;
                    }
                    bbox->idx = label_idx;
                    /* update bbox */
                    bbox->center_w = (_sigmoid(feature->center_w) * 2.0f - 0.5f + (float)ww) * (float)hnd->conf.strides[i];
                    bbox->center_h = (_sigmoid(feature->center_h) * 2.0f - 0.5f + (float)hh) * (float)hnd->conf.strides[i];
                    float sigmoid_w = _sigmoid(feature->len_w);
                    bbox->len_w = sigmoid_w * sigmoid_w * 4.0f * hnd->conf.anchors[i][cc][0] * (float)hnd->conf.strides[i];
                    float sigmoid_h = _sigmoid(feature->len_h);
                    bbox->len_h = sigmoid_h * sigmoid_h * 4.0f * hnd->conf.anchors[i][cc][1] * (float)hnd->conf.strides[i];
                    bbox++;
                    feature++;
                    box_cnt++;
                    if (box_cnt >= hnd->conf.max_det) {
                        TEST_PRINTF("warning: max_det = %d is too small\n", hnd->conf.max_det);
                        goto exit;
                    }
                }
            }
        }       
    }
exit:
    /* nms */
    bbox = (struct gesture_yolov5_bbox_t*)hnd->bbox;
    TEST_PRINTF("bbox before nms\n");
    for (int i = 0; i < box_cnt; i++) {
        TEST_PRINTF("(%f,%f,%f,%f), %f, %d\n", 
            bbox[i].center_h, bbox[i].center_w, bbox[i].len_h, bbox[i].len_w, 
            bbox[i].confidence, bbox[i].idx);
    }
    box_cnt = _nms(bbox, box_cnt, hnd->conf.iou_threshold);
    TEST_PRINTF("bbox after nms\n");
    for (int i = 0; i < box_cnt; i++) {
        TEST_PRINTF("(%f,%f,%f,%f), %f, %d\n", 
            bbox[i].center_h, bbox[i].center_w, bbox[i].len_h, bbox[i].len_w, 
            bbox[i].confidence, bbox[i].idx);
    }
    /* App needs scale to different size, so don't scale here */
#if 0
    /* convert bbox information to source image */
    box_cnt = gesture_yolov5_scale_coordinate(bbox, box_cnt, hnd->conf.det_h, hnd->conf.det_w, hnd->conf.src_h, hnd->conf.src_w);
    TEST_PRINTF("bbox after scale\n");
    for (int i = 0; i < box_cnt; i++) {
        TEST_PRINTF("(%f,%f,%f,%f), %f, %d\n", 
            bbox[i].center_h, bbox[i].center_w, bbox[i].len_h, bbox[i].len_w, 
            bbox[i].confidence, bbox[i].idx);
    }
#endif
    return box_cnt;
}

void gesture_yolov5_post_fini(gesture_yolov5_post_hnd_t hnd)
{
    if (hnd->bbox) {
        free(hnd->bbox);
    }
    free(hnd);
}

#pragma GCC pop_options