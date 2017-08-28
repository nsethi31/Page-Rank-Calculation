import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class PageRank {

	static int TOTAL = 0;  // Count total number of vertices in the graph

	public static class PageRankInitMapper
	extends Mapper<Object, Text, IntWritable, Text>{

		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException {

			String[] tokens = value.toString().split("\t");
			context.write(new IntWritable(Integer.parseInt(tokens[0])), 
					new Text(tokens[1]));  // pass graph structure to reducer
			TOTAL += 1;
		}
	}

	public static class PageRankInitReducer
	extends Reducer<IntWritable,Text,Text,Text> {

		public void reduce(IntWritable key, Iterable<Text> values,
				Context context
				) throws IOException, InterruptedException {
			for (Text value : values) {
				context.write(
						new Text(key.toString() + "," + String.valueOf(1.0 / TOTAL)),	// initialize rank=1/N
						value);															// pass graph structure
			}
		}
	}

	public static class PageRankMapper
	extends Mapper<Object, Text, IntWritable, Text>{

		public void map(Object key, Text value, Context context
				) throws IOException, InterruptedException {
			String[] ioTokens = value.toString().split("\t");
			String[] irTokens = ioTokens[0].split(",");
			String inlink = irTokens[0];
			String[] outlinks = ioTokens[1].split(",");
			Double rank = Double.parseDouble(irTokens[1]);
			for (String outlink : outlinks) 
				context.write(new IntWritable(Integer.parseInt(outlink)), 				// pass rank from inlink to
						new Text("a@" + String.valueOf(rank / outlinks.length)));		// to each outlink
			context.write(new IntWritable(Integer.parseInt(inlink)),
					new Text("b@" + ioTokens[1]));										// pass graph structure
		}
	}

	public static class PageRankReducer
	extends Reducer<IntWritable, Text, Text, Text> {

		double beta = 1;
		boolean isFinal;

		protected void setup(Context context) throws IOException {
			Configuration conf = context.getConfiguration();
			beta = conf.getDouble("beta", 1);
			isFinal = conf.getBoolean("isFinal", false);
		}

		public void reduce(IntWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {

			double rank = 0;
			String outlinks = "";

			for (Text val : values) {
				String[] tokens = val.toString().split("@");
				if (tokens[0].equals("a"))
					rank += Double.parseDouble(tokens[1]);
				else
					outlinks = tokens[1];
			}
			rank = rank * beta + (1 - beta) / TOTAL;
			if (isFinal)  // check if it is final round of iteration, we don't pass graph structure if it is
				context.write(new Text(key.toString()), new Text(String.valueOf(rank)));	// page<\t>rank
			else
				context.write(new Text(key.toString() + "," + String.valueOf(rank)), 		// page,rank<\t>outlinks
						new Text(outlinks));
		}
	}

	public static void main(String[] args) throws Exception {

		String inputPath = args[0];
		String outputPath = args[1];
		int maxIter = Integer.parseInt(args[2]);
		double beta = Double.parseDouble(args[3]);

//		String inputPath = "input_pagerank.txt";
//		String outputPath = "output";
//		int maxIter = 10;
//		double beta = 0.8;

		Configuration conf0 = new Configuration();
		Job job0 = Job.getInstance(conf0, "PageRankInit");
		job0.setJarByClass(PageRank.class);
		job0.setMapperClass(PageRankInitMapper.class);
		job0.setReducerClass(PageRankInitReducer.class);
		job0.setOutputKeyClass(IntWritable.class);
		job0.setOutputValueClass(Text.class);
		FileSystem.get(conf0).delete(new Path(outputPath), true);
		FileInputFormat.addInputPath(job0, new Path(inputPath));
		FileOutputFormat.setOutputPath(job0, new Path(outputPath, "0"));
		job0.waitForCompletion(true);

		for (int i = 0; i < maxIter; i++) {
			Configuration conf = new Configuration();
			conf.setDouble("beta", beta);
			conf.setBoolean("isFinal", (i + 1 == maxIter ? true : false));
			Job job = Job.getInstance(conf, "PageRank");
			job.setJarByClass(PageRank.class);
			job.setMapperClass(PageRankMapper.class);
			job.setReducerClass(PageRankReducer.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job, new Path(outputPath, String.valueOf(i)));
			FileOutputFormat.setOutputPath(job, new Path(outputPath, String.valueOf(i + 1)));
			job.waitForCompletion(true);
		}
	}

}