/*!
 * Add negative label to softmax, currently only support single_output
 * \file softmax_with_neg_output-inl.h
 * \brief
 * \author Yinghong Li
*/
#ifndef MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace softmax_with_neg_enum {
enum SoftmaxWithNegativeOutputOpInputs {kData, kLabel};
enum SoftmaxWithNegativeOutputOpOutputs {kOut};
enum SoftmaxWithNegativeOutputNormType {kNull, kBatch, kValid};
enum SoftmaxWithNegativeOutputOpResource {kTempSpace};
}  // namespace softmax_with_neg_enum

struct SoftmaxWithNegativeOutputParam : public dmlc::Parameter<SoftmaxWithNegativeOutputParam> {
  float grad_scale;
  float neg_grad_scale;
  float ignore_label;
  bool use_ignore;
  bool preserve_shape;
  int normalization;
  bool out_grad;
  bool ignore_negative;
  DMLC_DECLARE_PARAMETER(SoftmaxWithNegativeOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(neg_grad_scale).set_default(1.0f)
    .describe("Scale the gradienti of negative samples by a float factor (grad_scale * neg_grad_scale)");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("the label value will be ignored during backward (only works if "
      "use_ignore is set to be true).");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to true, the ignore_label value will not contribute "
      "to the backward gradient");
    DMLC_DECLARE_FIELD(preserve_shape).set_default(false)
    .describe("If true, for a (n_1, n_2, ..., n_d, k) dimensional "
      "input tensor, softmax will generate (n1, n2, ..., n_d, k) output, "
      "normalizing the k classes as the last dimension.");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", softmax_with_neg_enum::kNull)
    .add_enum("batch", softmax_with_neg_enum::kBatch)
    .add_enum("valid", softmax_with_neg_enum::kValid)
    .set_default(softmax_with_neg_enum::kNull)
    .describe("If set to null, op will do nothing on output gradient."
              "If set to batch, op will normalize gradient by divide batch size"
              "If set to valid, op will normalize gradient by divide sample not ignored");
    DMLC_DECLARE_FIELD(out_grad)
    .set_default(false)
    .describe("Apply weighting from output gradient");
    DMLC_DECLARE_FIELD(ignore_negative)
    .set_default(false)
    .describe("If set to true, will ignore all samples with negative label and perform normal softmax");
  };
};

template<typename xpu, typename DType>
class SoftmaxWithNegativeOutputOp : public Operator {
 public:
  explicit SoftmaxWithNegativeOutputOp(SoftmaxWithNegativeOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "SoftmaxWithNegativeOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1) << "SoftmaxWithNegativeOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.preserve_shape) {
      Tensor<xpu, 2, DType> data = in_data[softmax_with_neg_enum::kData].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = out_data[softmax_with_neg_enum::kOut].FlatTo2D<xpu, DType>(s);
      Softmax(out, data);
    } else {
      int n = in_data[softmax_with_neg_enum::kData].size(0);
      int k = in_data[softmax_with_neg_enum::kData].Size()/n;
      Shape<2> s2 = Shape2(n, k);
      Tensor<xpu, 2, DType> data =
         in_data[softmax_with_neg_enum::kData].get_with_shape<xpu, 2, DType>(s2, s);
      Tensor<xpu, 2, DType> out =
         out_data[softmax_with_neg_enum::kOut].get_with_shape<xpu, 2, DType>(s2, s);
      Softmax(out, data);
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_grad.size(), 1);
    CHECK_GE(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Shape<1> label_shape = Shape1(in_data[softmax_with_neg_enum::kLabel].Size());
    Shape<2> data_shape;
    if (param_.preserve_shape) {
      data_shape = out_data[softmax_with_neg_enum::kOut].shape_.FlatTo2D();
    } else {
      int n = out_data[softmax_with_neg_enum::kOut].size(0);
      data_shape = Shape2(n, out_data[softmax_with_neg_enum::kOut].Size()/n);
    }
    Tensor<xpu, 1, DType> label = in_data[softmax_with_neg_enum::kLabel].get_with_shape<xpu, 1, DType>(
          label_shape, s);
    Tensor<xpu, 2, DType> out =
          out_data[softmax_with_neg_enum::kOut].get_with_shape<xpu, 2, DType>(data_shape, s);
    Tensor<xpu, 2, DType> grad =
          in_grad[softmax_with_neg_enum::kData].get_with_shape<xpu, 2, DType>(data_shape, s);
    index_t valid_cnt = label.shape_.Size();
    if (param_.use_ignore) {
        SoftmaxWithNegativeGrad(grad, out, label, DType(param_.neg_grad_scale), static_cast<DType>(param_.ignore_label), param_.ignore_negative);
    } else {
        SoftmaxWithNegativeGrad(grad, out, label, DType(param_.neg_grad_scale), param_.ignore_negative);
    }
    if (param_.normalization == softmax_with_neg_enum::kBatch) {
        valid_cnt = label.size(0);
    } else if (param_.normalization == softmax_with_neg_enum::kValid) {
        int i_label = static_cast<int>(param_.ignore_label);
        Tensor<cpu, 1, DType> workspace =
          ctx.requested[softmax_with_neg_enum::kTempSpace].get_host_space_typed<1, DType>(
          label.shape_);
        Copy(workspace, label, label.stream_);
        for (index_t i = 0; i < label.size(0); ++i) {
          if (static_cast<int>(workspace[i]) == i_label || static_cast<int>(workspace[i]) == -i_label) {
            valid_cnt--;
          }
        }
        valid_cnt = valid_cnt == 0 ? 1 : valid_cnt;
    } else {
      valid_cnt = 1;
    }
    grad *= DType(param_.grad_scale / valid_cnt);
    if (param_.out_grad) {
      Tensor<xpu, 2, DType> ograd =
        out_grad[softmax_with_neg_enum::kOut].get_with_shape<xpu, 2, DType>(data_shape, s);
      grad *= ograd;
    }
  }

 private:
  SoftmaxWithNegativeOutputParam param_;
};  // class SoftmaxWithNegativeOutputOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxWithNegativeOutputParam param, int dtype);

#if DMLC_USE_CXX11
class SoftmaxWithNegativeOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    
    // do NOT allow probabilistic labels, only use single integer label
    SHAPE_ASSIGN_CHECK(*in_shape, softmax_with_neg_enum::kLabel, Shape1(dshape[0]));
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxWithNegativeOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxWithNegativeOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.out_grad) {
      return {in_data[softmax_with_neg_enum::kLabel], out_data[softmax_with_neg_enum::kOut],
              out_grad[softmax_with_neg_enum::kOut]};
    } else {
      return {in_data[softmax_with_neg_enum::kLabel], out_data[softmax_with_neg_enum::kOut]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[softmax_with_neg_enum::kOut], in_grad[softmax_with_neg_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmax_with_neg_enum::kData], out_data[softmax_with_neg_enum::kOut]}};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  SoftmaxWithNegativeOutputParam param_;
};  // class SoftmaxWithNegativeOutputProp

#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
