/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.cloudml.zen.ml.sampler

import java.util.Random

import spire.math.{Numeric => spNum}


class ScalingSampler(implicit ev: spNum[Double])
  extends Sampler[Double] {
  private var scale: Double = _
  private var sampler: Sampler[_] = _

  protected def numer: spNum[Double] = ev

  def apply(state: Int): Double = sampler.applyDouble(state) * scale

  def norm: Double = sampler.normDouble * scale

  def sampleFrom(base: Double, gen: Random): Int = sampler.sampleFromDouble(base / scale, gen)

  def resampleFrom(base: Double, gen: Random, state: Int): Int = sampler.resampleFromDouble(base / scale, gen, state)

  def resetScaling(scale: Double, sampler: Sampler[_]): ScalingSampler = {
    this.scale = scale
    this.sampler = sampler
    this
  }
}
