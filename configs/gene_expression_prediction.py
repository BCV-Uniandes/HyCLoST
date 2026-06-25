# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from meru.config import LazyCall as L
from meru.evaluation.prediction import GeneExpressionPrediction

evaluator = L(GeneExpressionPrediction)(
    image_size=224,
    num_workers=0,
)
