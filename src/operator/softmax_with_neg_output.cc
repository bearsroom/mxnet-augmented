/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax_with_neg_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxWithNegativeOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxWithNegativeOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxWithNegativeOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SoftmaxWithNegativeOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxWithNegativeOutput, SoftmaxWithNegativeOutputProp)
.describe("Perform a softmax transformation on input, backprop with logloss.")
.add_argument("data", "Symbol", "Input data to softmax.")
.add_argument("label", "Symbol", "Label data, can also be "\
              "probability value with same shape as data")
.add_arguments(SoftmaxWithNegativeOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
