/*
 * MXNET implementation of center loss
 * Ref: A Discriminative Feature Learning Approach for Deep Face Recognition
 */


#ifndef MXNET_OPERATOR_CENTER_LOSS_INL_H_
#define MXNET_OPERATOR_CENTER_LOSS_INL_H_

#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace centerloss {
enum CenterLossOpInputs {kData, kLabel};
enum CenterLossOpOutputs {kOut, kCenterDiff};
enum CenterLossOpAuxiliary {kCenterVec, kLabelCount};
enum CenterLossOpResource {kTempSpace};
} // centerloss

struct CenterLossParam : public dmlc::Parameter<CenterLossParam> {
  float grad_scale;
  float center_update_scale;
  int num_classes;
  bool allow_negative_label;
  DMLC_DECLARE_PARAMETER(CenterLossParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scale the gradient by a float factor");
    DMLC_DECLARE_FIELD(center_update_scale).set_default(0.5f)
    .describe("Scale the update value of center vectors by a float factor");
    DMLC_DECLARE_FIELD(num_classes).set_lower_bound(1)
    .describe("Number of classes (number of center vectors).");
    DMLC_DECLARE_FIELD(allow_negative_label).set_default(false)
    .describe("Allow negative label, i.e. label a negative example of a class will be expected as -(label_idx), do NOT support label with index 0");
  };
};


template<typename xpu, typename DType>
class CenterLossOp: public Operator {
  public:
    explicit CenterLossOp(CenterLossParam param) : param_(param) {}

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_states) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_GE(in_data.size(), 2) << "CenterLoss Input: [data, label]";
      CHECK_EQ(out_data.size(), 2) << "CenterLoss Output: [output, center_diff]";
      CHECK_EQ(aux_states.size(), 2) << "CenterLoss Auxiliary states: [center_vec, label_count]";
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TShape& ishape = in_data[centerloss::kData].shape_;

      Tensor<xpu, 2, DType> data = in_data[centerloss::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
      Tensor<xpu, 1, DType> label = in_data[centerloss::kLabel].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = out_data[centerloss::kOut].get<xpu, 1, DType>(s);
      Tensor<xpu, 2, DType> diff = out_data[centerloss::kCenterDiff].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> center_vec = aux_states[centerloss::kCenterVec].get<xpu, 2, DType>(s);

      // calculate distance between input vector and center vector
      Tensor<cpu, 1, DType> workspace = ctx.requested[centerloss::kTempSpace].get_host_space_typed<1, DType>(
                                        label.shape_);
      Copy(workspace, label, label.stream_);
      Tensor<xpu, 1, DType> count = aux_states[centerloss::kLabelCount].get<xpu, 1, DType>(s);
      Tensor<cpu, 1, DType> count_cpu = ctx.requested[centerloss::kTempSpace].get_host_space_typed<1, DType>(
                                        count.shape_);
      Copy(count_cpu, count, count.stream_);
      count_cpu = DType(0.0f);

      if (param_.allow_negative_label) {
        Tensor<cpu, 1, DType> abs_label = ctx.requested[centerloss::kTempSpace].get_host_space_typed<1, DType>(label.shape_);
        abs_label = F<mshadow_op::abs>(workspace);
        for (index_t i = 0; i < ishape[0]; i++) {
          index_t gt_label = static_cast<int>(abs_label[i]);
          diff[i] = data[i] - center_vec[gt_label];
          count_cpu[gt_label] += DType(1.0f);
        }
        diff *= broadcast<0>(F<mshadow_op::ge_zero>(label), diff.shape_);
      } else { 
        for (index_t i = 0; i < ishape[0]; i++) {
          index_t gt_label = static_cast<int>(workspace[i]);
          diff[i] = data[i] - center_vec[gt_label];
          count_cpu[gt_label] += DType(1.0f);
        }
      }
      out = sumall_except_dim<0>(F<mshadow_op::square>(diff));
      out = F<mshadow_op::square_root>(out);
      Copy(count, count_cpu, count.stream_);
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_states) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(in_data.size(), 2) << "CenterLoss Input: [data, label]";
      CHECK_EQ(out_data.size(), 2);
      CHECK_GE(in_grad.size(), 1) << "in_grad size invalid, expected 1, " << in_grad.size() << " given";
      CHECK_GE(req.size(), 1) << "req size invalid, expected 1, " << req.size() << "given";
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Tensor<xpu, 1, DType> label = in_data[centerloss::kLabel].get<xpu, 1, DType>(s);
      Tensor<xpu, 2, DType> diff = out_data[centerloss::kCenterDiff].get<xpu, 2, DType>(s);
      const TShape& ishape = in_grad[centerloss::kData].shape_;

      // gradient of data
      Tensor<xpu, 2, DType> grad = in_grad[centerloss::kData].get_with_shape<xpu, 2, DType>(
          Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s); 
      grad = diff;
      grad *= DType(param_.grad_scale);

      // update center_vec
      Tensor<xpu, 2, DType> center_vec = aux_states[centerloss::kCenterVec].get<xpu, 2, DType>(s);
      Tensor<cpu, 1, DType> workspace = ctx.requested[centerloss::kTempSpace].get_host_space_typed<1, DType>(
                                        label.shape_);
      Copy(workspace, label, label.stream_);
      Tensor<xpu, 1, DType> count = aux_states[centerloss::kLabelCount].get<xpu, 1, DType>(s);

      Tensor<xpu, 2, DType> delta_c = ctx.requested[centerloss::kTempSpace].get_space_typed<xpu, 2, DType>(Shape2(param_.num_classes, ishape.ProdShape(1, ishape.ndim())), s);
      delta_c = DType(0.0f);

      if (param_.allow_negative_label) {
        Tensor<cpu, 1, DType> abs_label = ctx.requested[centerloss::kTempSpace].get_host_space_typed<1, DType>(label.shape_);
        abs_label = F<mshadow_op::abs>(workspace);
        for (index_t i = 0; i < ishape[0]; i++) {
          index_t gt_label = static_cast<int>(abs_label[i]);
          delta_c[gt_label] += diff[i];
        }
      } else {
        for (index_t i = 0; i < ishape[0]; i++) {
          index_t gt_label = static_cast<int>(workspace[i]);
          delta_c[gt_label] += diff[i];
        }
      }
      delta_c /= broadcast<0>(F<mshadow::op::plus>(count, DType(1.0f)), delta_c.shape_);
      delta_c *= DType(param_.center_update_scale);
      center_vec += delta_c;
    }

  private:
    CenterLossParam param_;

}; // class CenterLossOp


// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateCenterLossOp(CenterLossParam param, int dtype);

#if DMLC_USE_CXX11
class CenterLossProp : public OperatorProperty {
  public:
    std::vector<std::string> ListArguments() const override {
      return {"data", "label"};
    }

    std::vector<std::string> ListOutputs() const override {
      return {"output", "center_diff"};
    }

    std::vector<std::string> ListAuxiliaryStates() const override {
      return {"center_vec", "label_count"};
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
      const TShape &dshape = (*in_shape)[centerloss::kData];
      // require data to be known
      if (dshape.ndim() == 0) return false;

      index_t num_input = dshape.ProdShape(1, dshape.ndim());
      auto &lshape = (*in_shape)[centerloss::kLabel];
      if (lshape.ndim() == 0) {
        lshape = Shape1(dshape[0]);
      } else {
        const TShape inferred_lshape = Shape1(dshape[0]);
        if (lshape != inferred_lshape) {
          std::ostringstream os;
          os << "Shape inconsistent, Provided " << '=' << lshape << ','
             << " inferred shape=" << inferred_lshape;
          throw ::mxnet::op::InferShapeError(os.str(), 1);
        }
      }
      out_shape->clear();
      out_shape->push_back(Shape1(dshape[0]));
      out_shape->push_back(Shape2(dshape[0], num_input));
      aux_shape->clear();
      aux_shape->push_back(Shape2(param_.num_classes, num_input));
      aux_shape->push_back(Shape1(param_.num_classes));
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
      out_type->push_back(dtype);
      aux_type->clear();
      aux_type->push_back(dtype);
      aux_type->push_back(dtype);
      return true;
    }

    OperatorProperty* Copy() const override {
      auto ptr = new CenterLossProp();
      ptr->param_ = param_;
      return ptr;
    }

    std::string TypeString() const override {
      return "CenterLoss";
    }

    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {in_data[centerloss::kLabel], out_data[centerloss::kCenterDiff]};
    }

    std::vector<ResourceRequest> ForwardResource(
        const std::vector<TShape> &in_shape) const override {
      return {ResourceRequest::kTempSpace};
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
    CenterLossParam param_;
}; // class CenterLossProp
#endif  // DMLC_USE_CXX11

} // namespace op
} // namespace mxnet
#endif // MXNET_OPERATOR_CENTER_LOSS_INL_H_
