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

import java.io.IOException

import breeze.linalg._
import com.github.cloudml.zen.ml.semiSupervised.GLDADefines._
import com.github.cloudml.zen.ml.util.{LoaderUtils, SparkUtils}
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
    val saveAsSolid = sc.getConf.get(cs_saveAsSolid).toBoolean
    val savPath = if (saveAsSolid) new Path(path + ".sav") else new Path(path)
    val savDir = savPath.toUri.toString
    val metaDir = LoaderUtils.metadataPath(savDir)
    val dataDir = LoaderUtils.dataPath(savDir)
    val fs = SparkUtils.getFileSystem(sc.getConf, savPath)
    fs.delete(savPath, true)

    val metadata = toStringMeta
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

  def toStringMeta: String = {
    val HyperParams(eta, mu, alpha) = params
    val json = ("class" -> sv_className) ~ ("version" -> sv_formatVersion) ~
      ("numTopics" -> numTopics) ~ ("numGroups" -> numGroups) ~ ("numTerms" -> numTerms) ~
      ("eta" -> eta) ~ ("mu" -> mu) ~ ("alpha" -> alpha)
    compact(render(json))
  }

  override protected def formatVersion: String = sv_formatVersion
}

object GLDAModel extends Loader[DistributedGLDAModel] {
  case class MetaT(clazz: String, version: String, numTopics: Int, numGroups: Int, numTerms: Int,
    eta: Float, mu: Float, alpha: Float)

  override def load(sc: SparkContext, path: String): DistributedGLDAModel = {
    val saveAsSolid = sc.getConf.get(cs_saveAsSolid).toBoolean
    val (metas, rdd) = if (saveAsSolid) {
      LoaderUtils.HDFSFile2RDD(sc, path, parseMeta, parseLine)
    } else {
      val metas = parseMeta(sc.textFile(LoaderUtils.metadataPath(path)).first())
      val rdd = sc.textFile(LoaderUtils.dataPath(path)).map(parseLine(metas, _))
      (metas, rdd)
    }
    val MetaT(clazz, version, numTopics, numGroups, numTerms, eta, mu, alpha) = metas
    validateSave(clazz, version)

    val partRdd = sc.getConf.getOption(cs_numPartitions).map(_.toInt) match {
      case Some(numParts) if rdd.partitions.length < numParts =>
        rdd.coalesce(numParts, shuffle=true)
      case _ => rdd
    }
    val storageLevel = StorageLevel.MEMORY_AND_DISK
    val termTopicsRDD = partRdd.persist(storageLevel)
    termTopicsRDD.count()
    val params = HyperParams(eta, mu, alpha)
    new DistributedGLDAModel(termTopicsRDD, numTopics, numGroups, numTerms, params, storageLevel)
  }

  def parseMeta(metadata: String): MetaT = {
    implicit val formats = DefaultFormats
    val json = parse(metadata)
    val clazz = (json \ "class").extract[String]
    val version = (json \ "version").extract[String]
    val numTopics = (json \ "numTopics").extract[Int]
    val numGroups = (json \ "numGroups").extract[Int]
    val numTerms = (json \ "numTerms").extract[Int]
    val eta = (json \ "eta").extract[Float]
    val mu = (json \ "mu").extract[Float]
    val alpha = (json \ "alpha").extract[Float]
    MetaT(clazz, version, numTopics, numGroups, numTerms, eta, mu, alpha)
  }

  def parseLine(metas: MetaT, line: String): (Int, Vector[Int]) = {
    val sv: Vector[Int] = SparseVector.zeros[Int](metas.numTopics)
    val arr = line.split("\t").view
    arr.tail.foreach { sub =>
      val Array(index, value) = sub.split(":")
      sv(index.toInt) = value.toInt
    }
    (arr.head.toInt, sv)
  }

  def validateSave(clazz: String, version: String): Unit = {
    if (clazz != sv_className || version != sv_formatVersion) {
      throw new Exception(s"GLDAModel.load did not recognize model with (className, format version):" +
        s"($clazz, $version). Supported: ($sv_className, $sv_formatVersion)")
    }
  }
}
