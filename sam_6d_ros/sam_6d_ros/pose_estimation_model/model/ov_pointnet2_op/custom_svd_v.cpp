// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define DEBUG_FLAG false

#include "custom_svd_v.hpp"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace TemplateExtension;

//! [op:ctor]
CustomSVDv::CustomSVDv(const ov::Output<ov::Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomSVDv::validate_and_infer_types() {
    // Support arbitrary batch dimensions, the last two dimensions are (M, N)
    const auto& svd_input = input(0);
    auto svd_shape = svd_input.get_partial_shape();
    set_output_type(0, get_input_element_type(0), ov::PartialShape(svd_shape)); // V

    // const auto& svd_input = input(0);
    // auto svd_shape = svd_input.get_partial_shape();
    // auto rank = svd_shape.rank().is_static() ? svd_shape.rank().get_length() : 0;
    // if (rank < 2) {
    //     throw std::runtime_error("CustomSVDv input must have at least 2 dimensions (batch..., M, N)");
    // }
    // auto elem_type = get_input_element_type(0);
    // // batch shape
    // std::vector<ov::Dimension> batch_dims;
    // for (size_t i = 0; i < rank - 2; ++i) batch_dims.push_back(svd_shape[i]);
    // auto m = svd_shape[rank - 2];
    // auto n = svd_shape[rank - 1];
    // // U: (batch..., M, M), S: (batch..., min(M,N)), V: (batch..., N, N)
    // std::vector<ov::Dimension> u_shape = batch_dims; u_shape.push_back(m); u_shape.push_back(m);
    // std::vector<ov::Dimension> s_shape = batch_dims;
    // // Fix: ov::Dimension does not have min, need to manually check
    // if (m.is_static() && n.is_static()) {
    //     s_shape.push_back(std::min(m.get_length(), n.get_length()));
    // } else {
    //     // When dynamic, use m conservatively
    //     s_shape.push_back(m);
    // }
    // std::vector<ov::Dimension> v_shape = batch_dims; v_shape.push_back(n); v_shape.push_back(n);
    // // set_output_type(0, elem_type, ov::PartialShape(u_shape)); // U
    // // set_output_type(1, elem_type, ov::PartialShape(s_shape)); // S
    // set_output_type(2, elem_type, ov::PartialShape(v_shape)); // V
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomSVDv::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomSVDv>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomSVDv::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

// Helper function to ensure proper SVD signs (similar to PyTorch)
void ensure_svd_v_signs(Eigen::MatrixXf& U, Eigen::VectorXf& S, Eigen::MatrixXf& V) {
    // Ensure singular values are non-negative and sorted in descending order
    for (int i = 0; i < S.size(); ++i) {
        if (S(i) < 0) {
            S(i) = -S(i);
            U.col(i) = -U.col(i);
        }
    }
    
    // Sort singular values in descending order
    std::vector<std::pair<float, int>> s_indices;
    for (int i = 0; i < S.size(); ++i) {
        s_indices.push_back({S(i), i});
    }
    std::sort(s_indices.begin(), s_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // Reorder U, S, V according to sorted singular values
    Eigen::MatrixXf U_new = U;
    Eigen::MatrixXf V_new = V;
    Eigen::VectorXf S_new = S;
    
    for (int i = 0; i < S.size(); ++i) {
        int old_idx = s_indices[i].second;
        S_new(i) = s_indices[i].first;
        U_new.col(i) = U.col(old_idx);
        V_new.col(i) = V.col(old_idx);
    }
    
    U = U_new;
    S = S_new;
    V = V_new;
}

//! [op:evaluate]
bool CustomSVDv::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (DEBUG_FLAG){
      std::cout << "======== [CPU ov_custom_svd_v] ======== " << std::endl;
    }
    // Support batch SVD, input shape: [batch..., M, N]
    const auto& in = inputs[0];
    auto shape = in.get_shape();
    if (shape.size() < 2)
        throw std::runtime_error("CustomSVDv input must have at least 2 dimensions");
    size_t rank = shape.size();
    size_t m = shape[rank - 2], n = shape[rank - 1];
    size_t batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) batch *= shape[i];
    const float* data = in.data<const float>();
    
    // Output shape
    // std::vector<size_t> u_shape = shape; u_shape[rank - 1] = m; // (batch..., M, M)
    // std::vector<size_t> s_shape = shape; s_shape.pop_back(); s_shape[rank - 2] = std::min(m, n); // (batch..., min(M,N))
    std::vector<size_t> v_shape = shape; v_shape[rank - 2] = n; v_shape[rank - 1] = n; // (batch..., N, N)
    // outputs[0].set_shape(u_shape);
    // outputs[1].set_shape(s_shape);
    // outputs[2].set_shape(v_shape);
    outputs[0].set_shape(v_shape);
    // float* u_data = outputs[0].data<float>();
    // float* s_data = outputs[1].data<float>();
    // float* v_data = outputs[2].data<float>();
    float* v_data = outputs[0].data<float>();
    size_t in_mat_size = m * n;
    size_t u_mat_size = m * m;
    size_t s_vec_size = std::min(m, n);
    size_t v_mat_size = n * n;
    
    for (size_t b = 0; b < batch; ++b) {
        const float* batch_data = data + b * in_mat_size;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(batch_data, m, n);
        
        // Use Eigen's BDCSVD for better numerical stability (similar to LAPACK)
        Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        // Get SVD results
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf V = svd.matrixV();
        
        // Ensure proper signs and ordering (similar to PyTorch)
        ensure_svd_v_signs(U, S, V);
        
        // Write outputs
        // Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(u_data + b * u_mat_size, m, m) = U;
        // Eigen::Map<Eigen::VectorXf>(s_data + b * s_vec_size, s_vec_size) = S;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(v_data + b * v_mat_size, n, n) = V;
    }
    
    // Debug: print CustomSVDv outputs
    // const bool debug = false; // true / false
    if (DEBUG_FLAG) {
        std::cout << "[CustomSVDv Debug] U: ";
        std::cout << "\n[CustomSVDv Debug] V: ";
        std::cout << std::endl;

        FILE* fp = fopen("output/ov_svd.txt", "a");
        if (fp) {
            fprintf(fp, "----- ov custom_svd (input) -----\n");
            for (size_t i = 0; i < batch * in_mat_size; ++i) fprintf(fp, "%f ", data[i]);
            fprintf(fp, "\n");
            
            // fprintf(fp, "----- ov custom_svd (U) -----\n");
            // for (size_t i = 0; i < batch * u_mat_size; ++i) fprintf(fp, "%f ", u_data[i]);
            // fprintf(fp, "\n");
            fprintf(fp, "----- ov custom_svd (V) -----\n");
            for (size_t i = 0; i < batch * v_mat_size; ++i) fprintf(fp, "%f ", v_data[i]);
            fprintf(fp, "\n");
            fclose(fp);
        } else {
            std::cerr << "[CustomSVDv Debug] Failed to open output/ov_svd.txt for writing!" << std::endl;
        }
    }
    
    return true;
}

bool CustomSVDv::has_evaluate() const {
    return true;
}
//! [op:evaluate]