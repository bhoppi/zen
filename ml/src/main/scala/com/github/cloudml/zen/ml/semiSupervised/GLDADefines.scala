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

package com.github.cloudml.zen.ml.semiSupervised

import breeze.linalg._
import com.github.cloudml.zen.ml.util.CompressedVector
import org.apache.spark.SparkConf

import scala.collection.mutable


object GLDADefines {
  case class HyperParams(var eta: Float, var mu: Float, var alpha: Float)

  case class IntWrapper(var value: Int)
  case class DocBow(docId: Long, docGrp: Int, docTerms: SparseVector[Int])
  case class DocRec(docId: Long, docGrp: IntWrapper, docData: Array[Int])
  case class TermRec(termId: Int, termData: Array[Int])
  case class DataBlock(termRecs: Array[TermRec], DocRecs: Array[DocRec])
  case class ParaBlock(routes: Array[Array[Int]], index: mutable.HashMap[Int, Int], attrs: Array[CompressedVector])
  case class ShippedAttrsBlock(termIds: Array[Int], termAttrs: Array[CompressedVector])
  case class GlobalVars(piGK: DenseMatrix[Float], nK: Array[Int], dG: DenseVector[Long])

  type OptionMap = Map[String, String]

  val sv_formatVersion = "1.0"
  val sv_className = "com.github.cloudml.zen.ml.semiSupervised.DistributedGLDAModel"
  val cs_numPartitions = "zen.glda.numPartitions"
  val cs_inputPath = "zen.glda.inputPath"
  val cs_outputpath = "zen.glda.outputPath"
  val cs_burninIter = "zen.glda.burninIter"
  val cs_sampleRate = "zen.glda.sampleRate"
  val cs_chkptInterval = "zen.glda.chkptInterval"
  val cs_evalMetric = "zen.glda.evalMetric"
  val cs_docGrouper = "zen.glda.docGrouper"
  val cs_saveInterval = "zen.glda.saveInterval"
  val cs_saveAsSolid = "zen.glda.saveAsSolid"
  val cs_labelsRate = "zen.glda.labelsRate"

  def setAppConfs(conf: SparkConf, options: OptionMap = Map()): SparkConf ={
    conf.set(cs_burninIter, options.getOrElse("burniniter", "10"))
    conf.set(cs_sampleRate, options.getOrElse("samplerate", "1.0"))
    conf.set(cs_chkptInterval, options.getOrElse("chkptinterval", "10"))
    conf.set(cs_evalMetric, options.getOrElse("evalmetric", "none"))
    conf.set(cs_docGrouper, options.getOrElse("docgrouper", "discrete"))
    conf.set(cs_saveInterval, options.getOrElse("saveinterval", "0"))
    conf.set(cs_saveAsSolid, options.getOrElse("saveassolid", "true"))
    conf.set(cs_labelsRate, options.getOrElse("labelsrate", "1.0"))
  }

  def registerKryoClasses(conf: SparkConf): Unit = {
    conf.registerKryoClasses(Array(
      classOf[Array[Int]],
      classOf[mutable.HashMap[Int, Int]],
      classOf[HyperParams],
      classOf[IntWrapper],
      classOf[DocBow], classOf[SparseVector[Int]],
      classOf[DocRec], classOf[TermRec],
      classOf[Array[Object]],
      classOf[CompressedVector],
      classOf[DenseMatrix[Float]], classOf[DenseVector[Int]], classOf[DenseVector[Long]]
    ))
  }

  def getDensed(bv: Vector[Int]): DenseVector[Int] = bv match {
    case v: DenseVector[Int] => v
    case v: SparseVector[Int] =>
      val used = v.used
      val index = v.index
      val data = v.data
      val arr = new Array[Int](v.length)
      var i = 0
      while (i < used) {
        arr(index(i)) = data(i)
        i += 1
      }
      new DenseVector(arr)
  }
}
