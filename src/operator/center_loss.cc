/*
 * MXNET implementation of center loss
 * Ref: A Discriminative Feature Learning Approach for Deep Face Recognition
 */


#include "./center_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateCenterLossOp<cpu>(CenterLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    return new CenterLossOp<cpu, DType>(param);
  })
  return op;
}

Operator *CenterLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateCenterLossOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(CenterLossParam);

MXNET_REGISTER_OP_PROPERTY(CenterLoss, CenterLossProp)
.describe("Apply center loss to input.")
.add_argument("data", "Symbol", "Input data to center loss calculation")
.add_arguments(CenterLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

