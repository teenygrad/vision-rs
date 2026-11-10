/*
 * Copyright 2026 Teenygrad
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// GPU kernels used by the YOLO26 model (attention, detection decode, loss).
pub mod kernels;
/// Training loss functions for YOLO26 (anchor assignment, CIoU, classification).
pub mod loss;
/// The YOLO26 model architecture: blocks, variants, and the assembled network.
pub mod yolo26;
