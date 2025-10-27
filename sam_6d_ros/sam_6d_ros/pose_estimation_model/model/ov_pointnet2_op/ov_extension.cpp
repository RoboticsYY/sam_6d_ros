#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "ball_query.hpp"
#include "custom_svd.hpp"
#include "custom_svd_u.hpp"
#include "custom_svd_v.hpp"
#include "custom_det.hpp"
#include "custom_debug_node.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::OpExtension<TemplateExtension::BallQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::BallQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVD>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVD>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVDu>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVDu>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVDv>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVDv>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomDet>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomDet>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomDebugNode>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomDebugNode>>(),
    }));
//! [ov_extension:entry_point]
// clang-format on