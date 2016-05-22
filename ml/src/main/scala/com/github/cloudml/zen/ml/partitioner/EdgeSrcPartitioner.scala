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

package com.github.cloudml.zen.ml.partitioner

import com.github.cloudml.zen.ml.clustering.LDADefines._
import org.apache.spark.HashPartitioner
import org.apache.spark.graphx2._
import org.apache.spark.graphx2.impl.GraphImpl
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag


class EdgeSrcPartitioner(val partitions: Int) extends HashPartitioner(partitions) {
  @inline def getKey(et: EdgeTriplet[_, _]): Long = et.srcId

  override def equals(other: Any): Boolean = other match {
    case edp: EdgeSrcPartitioner =>
      edp.numPartitions == numPartitions
    case _ =>
      false
  }
}

object EdgeSrcPartitioner {
  def partitionByESP[VD: ClassTag, ED: ClassTag](input: Graph[VD, ED],
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val conf = edges.context.getConf
    val numPartitions = conf.getInt(cs_numPartitions, edges.partitions.length)
    val esp = new EdgeSrcPartitioner(numPartitions)
    val newEdges = input.triplets.mapPartitions(_.map(et =>
      (esp.getKey(et), Edge(et.srcId, et.dstId, et.attr))
    )).partitionBy(esp).map(_._2)
    GraphImpl(input.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }
}
