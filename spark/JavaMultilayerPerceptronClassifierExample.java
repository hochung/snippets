package com.kryptnostic;

import java.io.File;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class JavaMultilayerPerceptronClassifierExample {
    public static void main( String[] args ) {
        SparkSession sparkSession = openSparkSession();
        String path = new File( JavaMultilayerPerceptronClassifierExample.class.getClassLoader().getResource( "perceptron.txt" ).getPath() )
                .getAbsolutePath();
        Dataset<Row> df = sparkSession.read().format( "libsvm" ).option( "numFeatures", "4" ).load( path );
        Dataset<Row>[] splits = df.randomSplit( new double[]{0.6, 0.4}, 1234L );
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];
        int[] layers = new int[] {4,5,4,3};
        
        System.out.println( "Dataframe Schema:" );
        df.printSchema();
        /**
         * Dataframe Schema:
         * root
         *  |-- label: double (nullable = true)
         *  |-- features: vector (nullable = true)
         */

        MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setMaxIter(100);
        
        MultilayerPerceptronClassificationModel model = trainer.fit(train);
        
        System.out.println( "Extract Param map:" );
        System.out.println( model.extractParamMap() );
        /**
         * Extract Param map:
         * { mlpc_0c1f65ad0a88-featuresCol: features, 
         * mlpc_0c1f65ad0a88-labelCol: label,
         * mlpc_0c1f65ad0a88-predictionCol: prediction }
         */

        Dataset<Row> result = model.transform( test );
        System.out.println( "Result Schema" );
        result.printSchema();
        /**
         * Result Schema
         * root
         * |-- label: double (nullable = true)
         * |-- features: vector (nullable = true)
         * |-- prediction: double (nullable = true)
         */
        String path2 = new File( JavaMultilayerPerceptronClassifierExample.class.getClassLoader().getResource( "trained_model" ).getPath() )
                .getAbsolutePath();

        try {
//            model.save( path2 );
            model.write().overwrite().save( path2 );
            /**
             * create "trained_model" folder in the path that stores the model in parquet 
             */
        } catch ( IOException e ) {
            e.printStackTrace();
        }
    }

   public static SparkSession openSparkSession() {
        SparkConf sparkConf = new SparkConf().setMaster( "local[8]" )
                .setAppName( "Multilayer Perceptron Example" )
                .set( "spark.sql.warehouse.dir", "file:////sparkWorkingDir" )
                .set( "spark.cassandra.connection.host", "127.0.0.1" )
                .set( "spark.cassandra.connection.port", Integer.toString( 9042 ) )
                .set( "spark.cassandra.connection.ssl.enabled",
                        String.valueOf( false ) );

        return SparkSession.builder().config( sparkConf ).getOrCreate();
    }    
}