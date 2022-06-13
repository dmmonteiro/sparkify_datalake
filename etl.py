import os
import argparse
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour, weekofyear, date_format
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, FloatType)

config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

os.environ['AWS_ACCESS_KEY_ID'] = config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = config.get(
    'AWS', 'AWS_SECRET_ACCESS_KEY')
MAX_MEMORY = '5g'


def create_spark_session():
    """Creates a Spark session"""

    spark = SparkSession \
        .builder \
        .appName('Sparkify') \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .config("spark.driver.memory", MAX_MEMORY) \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads:
      song data from input_data JSON files
    Creates:
      song schema, song_table, artists_table
    Writes:
      parquet files with the data of all created tables
       in the output_data directory
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    print(f'Song data path: {song_data}')

    songs_schema = StructType([
        StructField('num_songs', IntegerType(), True),
        StructField('artist_id', StringType(), True),
        StructField('artist_latitude', StringType(), True),
        StructField('artist_longitude', StringType(), True),
        StructField('artist_location', StringType(), True),
        StructField('artist_name', StringType(), True),
        StructField('song_id', StringType(), True),
        StructField('title', StringType(), True),
        StructField('duration', FloatType(), True),
        StructField('year', IntegerType(), True),
    ])

    # read song data file
    df = spark.read.json(song_data, songs_schema)
    df.show(5)

    # extract columns to create songs table
    songs_table = df[
        'song_id', 'title', 'artist_id', 'year', 'duration'
    ].drop_duplicates(subset=['song_id'])
    songs_table.show(5)

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').mode(
        'overwrite').parquet(f'{output_data}/songs/')

    # extract columns to create artists table
    artists_table = df[
        'artist_id', 'artist_name', 'artist_location',
        'artist_latitude', 'artist_longitude'
    ].drop_duplicates(subset=['artist_id'])
    artists_table.show(5)

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(f'{output_data}/artists/')


def process_log_data(spark, input_data, output_data, local_data=False):
    """
    Loads:
      log data from input_data JSON files
    Creates:
      users_table, time table, songplay table
    Writes:
      parquet files with the data of all created tables
       in the output_data directory
    """
    # get filepath to log data file
    path_to_json = '*.json' if local_data else '*/*/*.json'
    log_data = f'{input_data}/log_data/{path_to_json}'

    # read log data file
    df = spark.read.json(log_data)
    df.show(5)

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")
    df.show(5)

    # extract columns for users table
    users_table = df[
        'userId', 'firstName', 'lastName', 'gender', 'level'
    ].drop_duplicates(subset=['userId'])

    # write users table to parquet files
    users_table.show(5)
    users_table.write.mode("overwrite").parquet(output_data + 'users/')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x/1000)
    df = df.withColumn('timestamp', get_timestamp('ts'))

    # create datetime column from original timestamp column
    get_datetime = udf(
        lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

    # add start date and month to use when creating songplay
    df = df.withColumn('start_date', get_datetime('timestamp'))
    df = df.withColumn('month', month(df.start_date))
    df.show(5)

    # extract columns to create time table
    time_table = df.withColumn("hour", hour(df.start_date)) \
                   .withColumn("day", dayofmonth(df.start_date)) \
                   .withColumn("week", weekofyear(df.start_date)) \
                   .withColumn("month", df.month) \
                   .withColumn("year", year(df.start_date)) \
                   .withColumn("weekday", dayofweek(df.start_date)) \
                   .select([
                       "start_date", "hour",
                       "day", "week", "month", "year", "weekday"])

    # write time table to parquet files partitioned by year and month
    time_table.show(5)
    time_table.write.mode('overwrite').partitionBy('year', 'month') \
        .parquet(f'{output_data}/time/')

    # read in song data to use for songplays table
    songs_df = spark.read.parquet(f'{output_data}/songs').repartition('title')
    df = df.join(songs_df, (df.song == songs_df.title))

    # extract columns from joined song and log datasets to create songplays table
    songplay_table = df[
        'start_date', 'userId',
        'level', 'song_id',
        'artist_id', 'location',
        'userAgent', 'year',
        'month'
    ].withColumn('songplay_id', monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplay_table.show(5)
    songplay_table.write.mode("overwrite").partitionBy('year', 'month') \
        .parquet(f'{output_data}/songplay/')


def main():
    """
    Creates Spark session
    Gets input/output data paths (local or S3)
    Processes Song Data
    Processes Log Data
    """
    # parse script arguments
    parser = argparse.ArgumentParser(description="Process script arguments")
    parser.add_argument('--local', action='store_true')
    parser.set_defaults(local=False)
    args = parser.parse_args()

    spark = create_spark_session()

    # gets S3 keys if --local argument not provided
    key = 'LOCAL' if args.local else 'S3'
    input_data = config.get(key, 'INPUT_DATA')
    output_data = config.get(key, 'OUTPUT_DATA')

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data, local_data=args.local)


if __name__ == "__main__":
    main()
