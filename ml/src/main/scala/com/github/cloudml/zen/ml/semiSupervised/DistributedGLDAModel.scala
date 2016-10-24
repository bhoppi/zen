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

import java.io._

import breeze.linalg._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util._
import org.apache.hadoop.fs.{FileUtil, Path}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._


class DistributedGLDAModel(@transient val termTopicsRDD: RDD[(Int, Vector[Int])],
  val numTopics: Int,
  val numGroups: Int,
  val numTerms: Int,
  val params: HyperParams,
  val storageLevel: StorageLevel) extends Serializable with Saveable {
  def save(): Unit = {
    val sc = termTopicsRDD.context
    val outputPath = sc.getConf.get(cs_outputpath)
    save(sc, outputPath)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    val HyperParams(alpha, beta, eta, mu) = params
    val json = ("class" -> sv_className) ~ ("version" -> sv_formatVersion) ~
      ("numTopics" -> numTopics) ~ ("numGroups" -> numGroups) ~ ("numTerms" -> numTerms) ~
      ("alpha" -> alpha) ~ ("beta" -> beta) ~ ("eta" -> eta) ~ ("mu" -> mu)
    val metadata = compact(render(json))

    val saveAsSolid = sc.getConf.get(cs_saveAsSolid).toBoolean
    val savPath = if (saveAsSolid) new Path(path + ".sav") else new Path(path)
    val savDir = savPath.toUri.toString
    val metaDir = LoaderUtils.metadataPath(savDir)
    val dataDir = LoaderUtils.dataPath(savDir)
    val fs = SparkUtils.getFileSystem(sc.getConf, savPath)

    fs.delete(savPath, true)
    sc.parallelize(Seq(metadata), 1).saveAsTextFile(metaDir)
    // save model with the topic or word-term descending order
    termTopicsRDD.map { case (id, vector) =>
      val list = vector.activeIterator.filter(_._2 > 0).toSeq.sortBy(_._2).reverse
        .map(t => s"${t._1}:${t._2}").mkString("\t")
      s"$id\t$list"
    }.saveAsTextFile(dataDir)
    if (saveAsSolid) {
      val cpmgPath = new Path(path + ".cpmg")
      fs.delete(cpmgPath, true)
      var suc = fs.rename(new Path(metaDir + "/part-00000"), new Path(dataDir + "/_meta"))
      if (suc) {
        suc = FileUtil.copyMerge(fs, new Path(dataDir), fs, cpmgPath, false, sc.hadoopConfiguration, null)
      }
      if (suc) {
        suc = fs.rename(cpmgPath, new Path(path))
      }
      fs.delete(savPath, true)
      fs.delete(cpmgPath, true)
      if (!suc) {
        throw new IOException("Save model error!")
      }
    }
  }

  override protected def formatVersion: String = sv_formatVersion
}

object GLDAModel extends Loader[DistributedGLDAModel] {
  case class MetaT(numTopics: Int, numGroups: Int, numTerms: Int, params: HyperParams)

  override def load(sc: SparkContext, path: String): DistributedGLDAModel = {
    val (loadedClassName, version, metadata) = LoaderUtils.loadMetadata(sc, path)
    val dataPath = LoaderUtils.dataPath(path)
    if (loadedClassName == sv_className && version == sv_formatVersion) {
      val metas = parseMeta(metadata)
      var rdd = sc.textFile(dataPath).map(line => parseLine(metas, line))
      sc.getConf.getOption(cs_numPartitions).map(_.toInt).filter(_ > rdd.getNumPartitions).foreach { np =>
        rdd = rdd.coalesce(np, shuffle=true)
      }
      loadGLDAModel(metas, rdd)
    } else {
      throw new Exception(s"GLDAModel.load did not recognize model with (className, format version):" +
        s"($loadedClassName, $version). Supported: ($sv_className, $sv_formatVersion)")
    }
  }

  def loadFromSolid(sc: SparkContext, path: String): DistributedGLDAModel = {
    val (metas, rdd) = LoaderUtils.HDFSFile2RDD(sc, path, header => parseMeta(parse(header)), parseLine)
    loadGLDAModel(metas, rdd)
  }

  def parseMeta(metadata: JValue): MetaT = {
    implicit val formats = DefaultFormats
    val numTopics = (metadata \ "numTopics").extract[Int]
    val numGroups = (metadata \ "numGroups").extract[Int]
    val numTerms = (metadata \ "numTerms").extract[Int]
    val params = (metadata \ "params").extract[HyperParams]
    MetaT(numTopics, numGroups, numTerms, params)
  }

  def parseLine(metas: MetaT, line: String): (Int, SparseVector[Int]) = {
    val sv = SparseVector.zeros[Int](metas.numTopics)
    val arr = line.split("\t").view
    arr.tail.foreach { sub =>
      val Array(index, value) = sub.split(":")
      sv(index.toInt) = value.toInt
    }
    (arr.head.toInt, sv)
  }

  def loadGLDAModel(metas: MetaT, rdd: RDD[(Int, SparseVector[Int])]): DistributedGLDAModel = {
    val MetaT(numTopics, numGroups, numTerms, params) = metas
    val storageLevel = StorageLevel.MEMORY_AND_DISK
    val termTopicsRDD = rdd.asInstanceOf[RDD[(Int, Vector[Int])]].persist(storageLevel)
    new DistributedGLDAModel(termTopicsRDD, numTopics, numGroups, numTerms, params, storageLevel)
  }
}
