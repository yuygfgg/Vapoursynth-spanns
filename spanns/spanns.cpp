#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
    coeffs(cv::Rect(0, 0, half_width, half_height)).setTo(cv::Scalar(0));
    
    idwt2d(coeffs, *rec_lo, *rec_hi, details);
}

static void generate_mask(const float* src, int width, int height, float gamma, float* mask) {
    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat B1, B2;
    wavelet_transform(src_mat, "bior1.1", B1);
    wavelet_transform(src_mat, "coif1", B2);
    
    cv::Mat abs_B1, abs_B2;
    cv::absdiff(B1, cv::Scalar(0), abs_B1);
    cv::absdiff(B2, cv::Scalar(0), abs_B2);
    
    cv::Mat B;
    cv::max(abs_B1, abs_B2, B);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::dilate(B, B, kernel);
    
    std::vector<float> sorted_data;
    sorted_data.reserve(width * height);
    for(int i = 0; i < height; ++i) {
        const float* row = B.ptr<float>(i);
        sorted_data.insert(sorted_data.end(), row, row + width);
    }
    boost::sort::pdqsort(sorted_data.begin(), sorted_data.end());
    
    float thresh = sorted_data[static_cast<int>(gamma * (width * height - 1))];
    float min_val = sorted_data[0];
    
    if (thresh > min_val) {
        float scale = 1.0f / (thresh - min_val);
        float offset = -min_val * scale;
        B.convertTo(B, CV_32F, scale, offset);
        cv::min(B, cv::Scalar(1.0f), B); 
        cv::max(B, cv::Scalar(0.0f), B);
    } else {
        B = cv::Mat::zeros(B.size(), CV_32F);
    }
    
    memcpy(mask, B.ptr<float>(), width * height * sizeof(float));
}

class MPDistribution {
private:
    double beta_;
    double sigma_;
    double ratio_;
    
    std::pair<double, double> get_lambdas() const {
        double lambda_minus = beta_ * sigma_ * sigma_ * std::pow(1 - std::sqrt(ratio_), 2);
        double lambda_plus = beta_ * sigma_ * sigma_ * std::pow(1 + std::sqrt(ratio_), 2);
        return {lambda_minus, lambda_plus};
    }
    
    double cdf_aux_r(double x, double lambda_minus, double lambda_plus) const {
        return std::sqrt((lambda_plus - x) / (x - lambda_minus));
    }
    
    double cdf_aux_f(double x) const {
        auto [lambda_minus, lambda_plus] = get_lambdas();
        double var = beta_ * sigma_ * sigma_;
        
        if (x <= lambda_minus) return 0.0;
        if (x >= lambda_plus) return 1.0;
        
        double r = cdf_aux_r(x, lambda_minus, lambda_plus);
        
        double first_arctan;
        if (x == lambda_minus) {
            first_arctan = M_PI / 2;
        } else {
            first_arctan = std::atan((r * r - 1) / (2 * r));
        }
        
        double second_arctan;
        if (x == lambda_minus) {
            second_arctan = M_PI / 2;
        } else {
            second_arctan = std::atan((lambda_minus * r * r - lambda_plus) / (2 * var * (1 - ratio_) * r));
        }
        
        double sqrt_term = std::sqrt((lambda_plus - x) * (x - lambda_minus));
        
        return 1.0 / (2 * M_PI * ratio_) * (
            M_PI * ratio_ + 
            (1.0 / var) * sqrt_term - 
            (1 + ratio_) * first_arctan + 
            (1 - ratio_) * second_arctan
        );
    }

public:
    MPDistribution(double beta, double sigma, double ratio) 
        : beta_(beta), sigma_(sigma), ratio_(ratio) {}
    
    double cdf(double x) const {
        auto [lambda_minus, lambda_plus] = get_lambdas();
        
        if (x >= lambda_plus) return 1.0;
        if (x <= lambda_minus) return 0.0;
        
        double result = cdf_aux_f(x);
        
        if (ratio_ <= 1.0) {
            return result;
        }
        
        if (x >= lambda_minus && x <= lambda_plus) {
            result += (ratio_ - 1) / (2 * ratio_);
        }
        
        return std::clamp(result, 0.0, 1.0);
    }
};

static std::tuple<double, double, double> fit_mp_distribution(const Eigen::VectorXf& singular_values) {
    std::vector<double> sv(singular_values.data(), singular_values.data() + singular_values.size());
    
    boost::sort::pdqsort(sv.begin(), sv.end());
    
    double ub_init = sv[size_t(sv.size() * 0.9)];
    
    std::vector<double> filtered_sv;
    filtered_sv.reserve(sv.size());
    for(const auto& s : sv) {
        if(s < ub_init) {
            filtered_sv.push_back(s);
        }
    }
    
    size_t n = filtered_sv.size();
    double lm = filtered_sv[size_t(n * 0.05)];
    double lp = filtered_sv[size_t(n * 0.95)];
    
    double a = std::sqrt(lm);
    double b = std::sqrt(lp);
    
    double sigma = std::sqrt((a + b) * (a + b) / 4.0);
    double ratio = std::sqrt((b - a) / (a + b));
        
    return {1.0, sigma, ratio};
}

static void median_filter(const float* src, float* dst, int width, int height) {
    cv::Mat src_mat(height, width, CV_32F);
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::Mat dst_mat;
    cv::medianBlur(src_mat, dst_mat, 3);
    
    memcpy(dst, dst_mat.ptr<float>(), width * height * sizeof(float));
}

static void box_blur(const float* src, float* dst, int width, int height, float sigma) {
    cv::Mat src_mat(height, width, CV_32F);
    cv::Mat temp_mat;
    cv::Mat dst_mat;
    
    memcpy(src_mat.data, src, width * height * sizeof(float));
    
    cv::boxFilter(src_mat, temp_mat, -1, cv::Size(sigma, sigma), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    cv::boxFilter(temp_mat, dst_mat, -1, cv::Size(sigma, sigma), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    cv::boxFilter(dst_mat, temp_mat, -1, cv::Size(sigma, sigma), 
                cv::Point(-1,-1), true, cv::BORDER_DEFAULT);
    
    memcpy(dst, temp_mat.ptr<float>(), width * height * sizeof(float));
}

static void process_plane_spanns(const float* src, const float* ref, const float* dref, 
                                float* dst, int width, int height, float sigma, float tol, float gamma, int passes) {
    if (!src || !dst) return;
    
    std::vector<float> temp_ref(width * height);
    std::vector<float> temp_dref(width * height);
    std::vector<float> T(width * height);
    std::vector<float> mask(width * height);
    
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
    
    memcpy(T.data(), src, width * height * sizeof(float));
    
    for (int step = 0; step < passes; step++) {
        Eigen::MatrixXf T_mat = Eigen::Map<const Eigen::MatrixXf>(T.data(), height, width);
        Eigen::MatrixXf ref_mat = Eigen::Map<const Eigen::MatrixXf>(ref_ptr, height, width);

        Eigen::BDCSVD<Eigen::MatrixXf> svd_T(T_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXf noise = ref_mat - T_mat;
        Eigen::BDCSVD<Eigen::MatrixXf> svd_noise(noise, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::VectorXf S = svd_T.singularValues();
        Eigen::VectorXf S_noise = svd_noise.singularValues();

        auto [beta, sigma, ratio] = fit_mp_distribution(S_noise);
        MPDistribution mp_dist(beta, sigma, ratio);

        for (int i = 0; i < S.size(); i++) {
            double cdf_val = mp_dist.cdf(S(i));
            S(i) = S(i) - S(i) * (1.0 - cdf_val) * (1.0 - tol);
        }

        Eigen::MatrixXf result = svd_T.matrixU() * S.asDiagonal() * svd_T.matrixV().transpose();

        Eigen::Map<Eigen::MatrixXf>(T.data(), height, width) = result;
        
        for (int i = 0; i < width * height; i++) {
            T[i] = mask[i] * T[i] + (1.0f - mask[i]) * dref_ptr[i];
        }
    }
    
    for (int i = 0; i < width * height; i++) {
        float lb = std::min(src[i], ref_ptr[i]);
        float ub = std::max(src[i], ref_ptr[i]);
        dst[i] = std::clamp(T[i], lb, ub);
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