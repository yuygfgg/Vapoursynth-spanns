#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

typedef struct {
    VSNode *node;
    VSNode *ref1;
    VSNode *ref2;
    float sigma;
    float tol;
    float gamma;
    int passes;
    bool has_ref1;
    bool has_ref2;
} SpannsData;

// Stolen from pywt
namespace wavelets {
    const std::vector<double> bior1_1_dec_lo = {0.7071067811865476, 0.7071067811865476};
    const std::vector<double> bior1_1_dec_hi = {-0.7071067811865476, 0.7071067811865476};
    const std::vector<double> bior1_1_rec_lo = {0.7071067811865476, 0.7071067811865476};
    const std::vector<double> bior1_1_rec_hi = {0.7071067811865476, -0.7071067811865476};
    
    const std::vector<double> coif1_dec_lo = {-0.015655728135791993, -0.07273261951252645, 0.3848648468648578, 0.8525720202116004, 0.3378976624574818, -0.07273261951252645};
    const std::vector<double> coif1_dec_hi = {0.07273261951252645, 0.3378976624574818, -0.8525720202116004, 0.3848648468648578, 0.07273261951252645, -0.015655728135791993};
    const std::vector<double> coif1_rec_lo = {-0.07273261951252645, 0.3378976624574818, 0.8525720202116004, 0.3848648468648578, -0.07273261951252645, -0.015655728135791993};
    const std::vector<double> coif1_rec_hi = {-0.015655728135791993, 0.07273261951252645, 0.3848648468648578, -0.8525720202116004, 0.3378976624574818, 0.07273261951252645};
}

static void upsample(const std::vector<float>& input, std::vector<float>& output) {
    output.resize(input.size() * 2);
    for (size_t i = 0; i < input.size(); i++) {
        output[i * 2] = input[i];
        output[i * 2 + 1] = 0;
    }
}

static void conv1d(const std::vector<float>& input, const std::vector<double>& kernel,
                    std::vector<float>& output, int stride = 1) {
    int n = input.size();
    int m = kernel.size();
    output.resize((n + m - 1) / stride);
    
    for (size_t i = 0; i < output.size(); i++) {
        float sum = 0;
        for (int j = 0; j < m; j++) {
            int idx = i * stride + j;
            if (idx < n) {
                sum += input[idx] * kernel[j];
            }
        }
        output[i] = sum;
    }
}

static std::vector<float> periodic_pad(const std::vector<float>& input, int pad_size) {
    std::vector<float> padded(input.size() + 2 * pad_size);
    for (size_t i = 0; i < padded.size(); i++) {
        padded[i] = input[(i - pad_size + input.size()) % input.size()];
    }
    return padded;
}

static void dwt1d(const std::vector<float>& input, const std::vector<double>& dec_lo,
                    const std::vector<double>& dec_hi, std::vector<float>& cA, std::vector<float>& cD) {
    int pad_size = dec_lo.size() - 1;
    auto padded = periodic_pad(input, pad_size);
    
    conv1d(padded, dec_lo, cA, 2);
    conv1d(padded, dec_hi, cD, 2);
    
    int offset = pad_size / 2;
    if (offset > 0) {
        cA.erase(cA.begin(), cA.begin() + offset);
        cD.erase(cD.begin(), cD.begin() + offset);
        cA.resize(input.size() / 2);
        cD.resize(input.size() / 2);
    }
}

static void idwt1d(const std::vector<float>& cA, const std::vector<float>& cD,
                    const std::vector<double>& rec_lo, const std::vector<double>& rec_hi,
                    std::vector<float>& output) {
    std::vector<float> upsampled_cA, upsampled_cD;
    upsample(cA, upsampled_cA);
    upsample(cD, upsampled_cD);
    
    std::vector<float> rec_lo_conv, rec_hi_conv;
    conv1d(upsampled_cA, rec_lo, rec_lo_conv);
    conv1d(upsampled_cD, rec_hi, rec_hi_conv);
    
    int size = std::min(rec_lo_conv.size(), rec_hi_conv.size());
    output.resize(size);
    for (int i = 0; i < size; i++) {
        output[i] = rec_lo_conv[i] + rec_hi_conv[i];
    }
}

static void dwt2d(const cv::Mat& input, const std::vector<double>& dec_lo,
                    const std::vector<double>& dec_hi, cv::Mat& coeffs) {
    int height = input.rows;
    int width = input.cols;
    
    std::vector<float> row(width);
    std::vector<float> cA, cD;
    cv::Mat temp(height, width, CV_32F);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            row[j] = input.at<float>(i, j);
        }
        
        dwt1d(row, dec_lo, dec_hi, cA, cD);
        
        for (int j = 0; j < width/2; j++) {
            temp.at<float>(i, j) = cA[j];
            temp.at<float>(i, j + width/2) = cD[j];
        }
    }
    
    std::vector<float> col(height);
    coeffs = cv::Mat(height, width, CV_32F);
    
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            col[i] = temp.at<float>(i, j);
        }
        
        dwt1d(col, dec_lo, dec_hi, cA, cD);
        
        for (int i = 0; i < height/2; i++) {
            coeffs.at<float>(i, j) = cA[i];
            coeffs.at<float>(i + height/2, j) = cD[i];
        }
    }
}

static void idwt2d(const cv::Mat& coeffs, const std::vector<double>& rec_lo,
                    const std::vector<double>& rec_hi, cv::Mat& output) {
    int height = coeffs.rows;
    int width = coeffs.cols;
    
    std::vector<float> col_cA(height/2), col_cD(height/2);
    cv::Mat temp(height, width, CV_32F);
    
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height/2; i++) {
            col_cA[i] = coeffs.at<float>(i, j);
            col_cD[i] = coeffs.at<float>(i + height/2, j);
        }
        
        std::vector<float> reconstructed_col;
        idwt1d(col_cA, col_cD, rec_lo, rec_hi, reconstructed_col);
        
        for (int i = 0; i < height; i++) {
            temp.at<float>(i, j) = reconstructed_col[i];
        }
    }
    
    std::vector<float> row_cA(width/2), row_cD(width/2);
    output = cv::Mat(height, width, CV_32F);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width/2; j++) {
            row_cA[j] = temp.at<float>(i, j);
            row_cD[j] = temp.at<float>(i, j + width/2);
        }
        
        std::vector<float> reconstructed_row;
        idwt1d(row_cA, row_cD, rec_lo, rec_hi, reconstructed_row);
        
        for (int j = 0; j < width; j++) {
            output.at<float>(i, j) = reconstructed_row[j];
        }
    }
}

static void wavelet_transform(const cv::Mat& input, const char* wavelet_type, cv::Mat& details) {
    cv::Mat coeffs;
    const std::vector<double> *dec_lo, *dec_hi, *rec_lo, *rec_hi;

    if (strcmp(wavelet_type, "bior1.1") == 0) {
        dec_lo = &wavelets::bior1_1_dec_lo;
        dec_hi = &wavelets::bior1_1_dec_hi;
        rec_lo = &wavelets::bior1_1_rec_lo;
        rec_hi = &wavelets::bior1_1_rec_hi;
    } 
    else {
        dec_lo = &wavelets::coif1_dec_lo;
        dec_hi = &wavelets::coif1_dec_hi;
        rec_lo = &wavelets::coif1_rec_lo;
        rec_hi = &wavelets::coif1_rec_hi;
    }
    
    dwt2d(input, *dec_lo, *dec_hi, coeffs);
    
    int half_height = coeffs.rows / 2;
    int half_width = coeffs.cols / 2;
    cv::Mat roi = coeffs(cv::Rect(0, 0, half_width, half_height));
    roi = cv::Scalar(0);
    
    idwt2d(coeffs, *rec_lo, *rec_hi, details);
}

static void generate_mask(const float* src, int width, int height, float gamma, float* mask) {
    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat B1, B2;
    wavelet_transform(src_mat, "bior1.1", B1);
    wavelet_transform(src_mat, "coif1", B2);
    
    cv::absdiff(B1, cv::Scalar(0), B1);
    cv::absdiff(B2, cv::Scalar(0), B2);
    
    cv::Mat B;
    cv::max(B1, B2, B);
    
    cv::dilate(B, B, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    
    std::vector<float> sorted_data;
    sorted_data.reserve(width * height);
    for(int i = 0; i < height; ++i) {
        const float* row = B.ptr<float>(i);
        sorted_data.insert(sorted_data.end(), row, row + width);
    }
    std::sort(sorted_data.begin(), sorted_data.end());
    
    float thresh = sorted_data[static_cast<int>(gamma * (width * height - 1))];
    float min_val = sorted_data[0];
    
    if (thresh > min_val) {
        // (B - min_val) / (thresh - min_val)
        B.convertTo(B, CV_32F, 1.0/(thresh - min_val), -min_val/(thresh - min_val));
        cv::threshold(B, B, 1.0, 1.0, cv::THRESH_TRUNC);
    } else {
        B = cv::Mat::zeros(B.size(), CV_32F);
    }
    
    memcpy(mask, B.ptr<float>(), width * height * sizeof(float));
}

static std::tuple<double, double, double> fit_mp_distribution(const Eigen::VectorXf& singular_values) {
    std::vector<double> x(singular_values.data(), singular_values.data() + singular_values.size());
    size_t n = x.size();
    std::sort(x.begin(), x.end());
    
    double q05 = gsl_stats_quantile_from_sorted_data(&x[0], 1, n, 0.05);
    double q95 = gsl_stats_quantile_from_sorted_data(&x[0], 1, n, 0.95);
    
    double a = std::sqrt(q05);
    double b = std::sqrt(q95);
    
    double sigma = std::sqrt((a + b) * (a + b) / 4.0);
    double ratio = std::sqrt((b - a) / (a + b));
    
    return {1.0, sigma, ratio}; // beta = 1.0 for real matrices
}

struct mp_params {
    double beta;
    double sigma;
    double ratio;
};

static double mp_pdf(double x, void* params) {
    mp_params* p = (mp_params*)params;
    double beta = p->beta;
    double sigma = p->sigma;
    double ratio = p->ratio;
    
    double lambda_minus = beta * sigma * sigma * std::pow(1 - std::sqrt(ratio), 2);
    double lambda_plus = beta * sigma * sigma * std::pow(1 + std::sqrt(ratio), 2);

    if (x <= lambda_minus || x >= lambda_plus) {
        return 0.0;
    }

    double var = beta * sigma * sigma;
    double term1 = std::sqrt((lambda_plus - x) * (x - lambda_minus));
    double term2 = 2.0 * M_PI * ratio * var * x;
    
    return term1 / term2;
}

static double mp_cdf(double x, double beta, double sigma, double ratio) {
    if (ratio <= 0 || ratio >= 1) return 0.0;
    
    double lambda_minus = beta * sigma * sigma * std::pow(1 - std::sqrt(ratio), 2);
    double lambda_plus = beta * sigma * sigma * std::pow(1 + std::sqrt(ratio), 2);
    
    if (x >= lambda_plus) return 1.0;
    if (x <= lambda_minus) return 0.0;

    mp_params params = {beta, sigma, ratio};

    gsl_function F;
    F.function = &mp_pdf;
    F.params = &params;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
    double result, error;

    gsl_integration_qags(&F, lambda_minus, x, 0, 1e-9, 2000, w, &result, &error);
    
    gsl_integration_workspace_free(w);

    if (result < 0) result = 0;
    if (result > 1) result = 1;
    
    return result;
}

class MPDistribution {
private:
    double beta_;
    double sigma_;
    double ratio_;
    std::unordered_map<double, double> cdf_cache_;
    
public:
    MPDistribution(double beta, double sigma, double ratio) 
        : beta_(beta), sigma_(sigma), ratio_(ratio) {}
    
    double cdf(double x) {
        auto it = cdf_cache_.find(x);
        if (it != cdf_cache_.end()) {
            return it->second;
        }
        
        double result = mp_cdf(x, beta_, sigma_, ratio_);
        cdf_cache_[x] = result;
        return result;
    }
};

static void median_filter(const float* src, float* dst, int width, int height) {
    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat dst_mat;
    cv::medianBlur(src_mat, dst_mat, 3);
    
    memcpy(dst, dst_mat.ptr<float>(), width * height * sizeof(float));
}

static void box_blur(const float* src, float* dst, int width, int height, float sigma) {
    int kernel_size = std::ceil(sigma * 2);
    if (kernel_size % 2 == 0) kernel_size++;

    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat dst_mat;
    cv::boxFilter(src_mat, dst_mat, -1, cv::Size(kernel_size, kernel_size), 
                    cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    memcpy(dst, dst_mat.ptr<float>(), width * height * sizeof(float));
}

static void process_plane_spanns(const float* src, const float* ref, const float* dref, 
                                float* dst, int width, int height, float sigma, float tol, float gamma, int passes) {
    if (!src || !dst) return;
    
    std::vector<float> temp_ref(width * height);
    std::vector<float> temp_dref(width * height);
    std::vector<float> temp_buf(width * height);
    std::vector<float> mask(width * height);
    std::vector<float> orig_buf(width * height);
    
    memcpy(orig_buf.data(), src, width * height * sizeof(float));
    
    const float* ref_ptr = ref;
    const float* dref_ptr = dref;
    
    if (!ref_ptr) {
        median_filter(src, temp_ref.data(), width, height);
        ref_ptr = temp_ref.data();
    }
    
    if (!dref_ptr) {
        box_blur(src, temp_dref.data(), width, height, sigma);
        dref_ptr = temp_dref.data();
    }
    
    generate_mask(src, width, height, gamma, mask.data());
    
    memcpy(dst, src, width * height * sizeof(float));
    
    for (int step = 0; step < passes; step++) {
        Eigen::Map<const Eigen::MatrixXf> orig(orig_buf.data(), height, width);
        Eigen::Map<const Eigen::MatrixXf> dref_mat(dref_ptr, height, width);
        
        Eigen::MatrixXf noise = dref_mat - orig;
        
        Eigen::BDCSVD<Eigen::MatrixXf> svd(orig, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::BDCSVD<Eigen::MatrixXf> svd_noise(noise, Eigen::ComputeThinU | Eigen::ComputeThinV);
        
        Eigen::VectorXf S = svd.singularValues();
        Eigen::VectorXf S_noise = svd_noise.singularValues();
        
        auto [beta, sigma, ratio] = fit_mp_distribution(S_noise);
        MPDistribution mp_dist(beta, sigma, ratio);
            
        for (int i = 0; i < S.size(); i++) {
            double cdf_val = mp_dist.cdf(S(i));
            S(i) = S(i) - S(i) * (1.0 - cdf_val) * (1.0 - tol);
        }


        Eigen::MatrixXf result = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

        Eigen::Map<Eigen::MatrixXf>(temp_buf.data(), height, width) = result;

        for (int i = 0; i < width * height; i++) {
            float lb = std::min(orig_buf[i], ref_ptr[i]);
            float ub = std::max(orig_buf[i], ref_ptr[i]);
            dst[i] = std::clamp(temp_buf[i], lb, ub);
        }

        memcpy(orig_buf.data(), dst, width * height * sizeof(float));
    }

    for (int i = 0; i < width * height; i++) {
        dst[i] = mask[i] * dst[i] + (1.0f - mask[i]) * dref_ptr[i];
    }
}

static const VSFrame *VS_CC spannsGetFrame(int n, int activationReason, void *instanceData, 
                                         void **frameData, VSFrameContext *frameCtx, 
                                         VSCore *core, const VSAPI *vsapi) {
    SpannsData *d = (SpannsData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->has_ref1) vsapi->requestFrameFilter(n, d->ref1, frameCtx);
        if (d->has_ref2) vsapi->requestFrameFilter(n, d->ref2, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame *ref1 = d->has_ref1 ? vsapi->getFrameFilter(n, d->ref1, frameCtx) : nullptr;
        const VSFrame *ref2 = d->has_ref2 ? vsapi->getFrameFilter(n, d->ref2, frameCtx) : nullptr;
        
        VSFrame *dst = vsapi->newVideoFrame(vsapi->getVideoFrameFormat(src), 
                                            vsapi->getFrameWidth(src, 0), 
                                            vsapi->getFrameHeight(src, 0), 
                                            src, core);
        
        for (int p = 0; p < vsapi->getVideoFrameFormat(src)->numPlanes; p++) {
            const float *srcp = (const float *)vsapi->getReadPtr(src, p);
            const float *ref1p = ref1 ? (const float *)vsapi->getReadPtr(ref1, p) : nullptr;
            const float *ref2p = ref2 ? (const float *)vsapi->getReadPtr(ref2, p) : nullptr;
            float *dstp = (float *)vsapi->getWritePtr(dst, p);
            
            int plane_width = vsapi->getFrameWidth(src, p);
            int plane_height = vsapi->getFrameHeight(src, p);
            
            process_plane_spanns(srcp, ref1p, ref2p, dstp, 
                                plane_width, plane_height,
                                d->sigma, d->tol, d->gamma, d->passes);
        }
        
        vsapi->freeFrame(src);
        if (ref1) vsapi->freeFrame(ref1);
        if (ref2) vsapi->freeFrame(ref2);
        
        return dst;
    }
    return nullptr;

    (void) frameData;
}

static void VS_CC spannsFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    SpannsData *d = (SpannsData *)instanceData;
    vsapi->freeNode(d->node);
    if (d->ref1) vsapi->freeNode(d->ref1);
    if (d->ref2) vsapi->freeNode(d->ref2);
    free(d);

    (void) core;
}

static void VS_CC spannsCreate(const VSMap *in, VSMap *out, void *userData, 
                             VSCore *core, const VSAPI *vsapi) {
    SpannsData d;
    SpannsData *data;
    int err;
    
    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.ref1 = vsapi->mapGetNode(in, "ref1", 0, &err);
    d.has_ref1 = err != 1;
    d.ref2 = vsapi->mapGetNode(in, "ref2", 0, &err);
    d.has_ref2 = err != 1; 

    const VSVideoInfo *vi = vsapi->getVideoInfo(d.node);
    
    if (!vsh::isConstantVideoFormat(vi) || vi->format.sampleType != stFloat || 
        vi->format.bitsPerSample != 32 ||
        (vi->format.numPlanes != 1 && vi->format.numPlanes != 3)) {
        vsapi->mapSetError(out, "Spanns: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.sigma = (float)vsapi->mapGetFloat(in, "sigma", 0, &err);
    if (err) d.sigma = 1.0f;
    
    d.tol = (float)vsapi->mapGetFloat(in, "tol", 0, &err);
    if (err) d.tol = 0.7f;
    
    d.gamma = (float)vsapi->mapGetFloat(in, "gamma", 0, &err);
    if (err) d.gamma = 0.5f;
    
    d.passes = (int)vsapi->mapGetInt(in, "passes", 0, &err);
    if (err) d.passes = 2;
    
    data = (SpannsData *)malloc(sizeof(d));
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Spanns", vi, spannsGetFrame, spannsFree, 
                            fmParallel, deps, 1, data, core);

    (void) userData;
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.yuygfgg.spanns", "spanns", 
                        "SPANNS Denoising", VS_MAKE_VERSION(1, 0), 
                        VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SPANNS",
                            "clip:vnode;ref1:vnode:opt;ref2:vnode:opt;"
                            "sigma:float:opt;tol:float:opt;gamma:float:opt;"
                            "passes:int:opt;",
                            "clip:vnode;",
                            spannsCreate, nullptr, plugin);
}