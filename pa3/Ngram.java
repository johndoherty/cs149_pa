
import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class Ngram extends Configured implements Tool {

    public static class Map extends MapReduceBase implements Mapper<Text, Text, Text, Text> {

        private Text word = new Text();
        private Set<String> queryNgrams = new HashSet<String>();
        private int n;

        public void configure(JobConf job) {
            n = job.getInt("n", 1);
            try {
                Path[] cacheFiles = DistributedCache.getLocalCacheFiles(job);
                if (cacheFiles.length > 0) {
		    parseQueryFile(cacheFiles[0], n);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while getting cached files: " + StringUtils.stringifyException(ioe));
            }
        }

        private List<String> extractNgrams(String page, int n) {
            Tokenizer tokenizer = new Tokenizer(page);
            LinkedList<String> ngram = new LinkedList<String>();
            List<String> ngrams = new ArrayList<String>();
            while (tokenizer.hasNext()) {
                ngram.add(tokenizer.next());
                if (ngram.size() > n) {
                    ngram.remove();
                }
                String ngramString = "";
                for (String gram : ngram) {
                    ngramString += gram + " ";
                }
                ngrams.add(ngramString.trim());
            }
            return ngrams;
        }

        private void parseQueryFile(Path queryFile, int n) {
            try {
                BufferedReader fis = new BufferedReader(new FileReader(queryFile.toString()));
                String text = "";
                String line = null;
                while ((line = fis.readLine()) != null) {
                    text += line;
                }
                queryNgrams = new HashSet<String>(extractNgrams(text, n));
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file: "  + StringUtils.stringifyException(ioe));
            }
        }

        public void map(Text key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            System.out.println("Title: " + key.toString());
            int matchCount = 0;
            for (String ngram : extractNgrams(value.toString(), n)) {
                if (queryNgrams.contains(ngram)) {
                    matchCount++;
                }
            }
            output.collect(new Text(""), new Text(Integer.toString(matchCount) + " " + key.toString()));
        }
    }

    public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
            String bestArticle = "";
            int bestScore = 0;
            while (values.hasNext()) {
                String[] value = values.next().toString().split(" ");
                int count = Integer.parseInt(value[0]);
                String title = "";
                for (int i = 1; i < value.length; i++) {
                    title += value[i] + " ";
                }
                title = title.trim();
                if (count > bestScore || 
                (count == bestScore && title.compareTo(bestArticle) > 0)) {
                    bestArticle = title;
                    bestScore = count;
                }
            }
            output.collect(key, new Text(Integer.toString(bestScore) + " " + bestArticle));
        }
    }

    public static class PageRecordReader implements RecordReader<Text, Text> {
        private LineRecordReader lineReader;
        private LongWritable lineKey;
        private Text title;
        private Text lineValue;

        public PageRecordReader(JobConf job, FileSplit split) throws IOException {
            lineReader = new LineRecordReader(job, split);

            lineKey = lineReader.createKey();
            lineValue = lineReader.createValue();
            Text tmp = new Text();
            textToTitle(title, tmp);
        }

        private boolean textToTitle(Text nextTitle, Text value) throws IOException {
            String line = "";
            value.set("");
            String pattern = "<title>(.+?)</title>";
            if (!lineReader.next(lineKey, lineValue)) {
                return false;
            }
            line = lineValue.toString();

            while (!line.matches(pattern)) {
                value.append(lineValue.getBytes(), 0, lineValue.getLength());
                if (!lineReader.next(lineKey, lineValue)) {
                    return false;
                }
                line = lineValue.toString();
            }
            String title = line.replaceAll(pattern, "$1");
            System.err.println("Title: " + title);
            nextTitle.set(title);
            return true;
        }

        public boolean next(Text key, Text value) throws IOException {
            Text nextTitle = new Text();
            Text pageText = new Text();
            boolean hasNext = textToTitle(nextTitle, pageText);
            if (!hasNext) {
                return false;
            }

            key.set(title);
            value.set(pageText);
            title.set(nextTitle);
            return true;
        }

        public Text createKey() {
            return new Text("");
        }

        public Text createValue() {
            return new Text("");
        }

        public long getPos() throws IOException {
            return lineReader.getPos();
        }

        public void close() throws IOException {
            lineReader.close();
        }

        public float getProgress() throws IOException {
            return lineReader.getProgress();
        }
    }

    public static class PageFormat extends FileInputFormat<Text, Text> {
        public RecordReader<Text, Text> getRecordReader(InputSplit split, JobConf conf, Reporter reporter) throws IOException{
            reporter.setStatus(split.toString());
            return new PageRecordReader(conf, (FileSplit)split);
        }

	/*j@Override
	protected boolean isSplitable(FileSystem fs, Path filename) {
	    return false;
	}*/
    }

    public int run(String[] args) throws Exception {
        if (args.length != 4) {
            System.out.println("Invalid input");
            return 1;
        }
        int n = Integer.parseInt(args[0]);
        String queryFilePath = args[1];
        String inputPath = args[2];
        String outputPath = args[3];

        JobConf conf = new JobConf(getConf(), Ngram.class);
        conf.setJobName("ngram");

        conf.setOutputKeyClass(Text.class);
        conf.setOutputValueClass(Text.class);

        conf.setMapperClass(Map.class);
        conf.setCombinerClass(Reduce.class);
        conf.setReducerClass(Reduce.class);

        conf.setInputFormat(PageFormat.class);
        conf.setOutputFormat(TextOutputFormat.class);

        DistributedCache.addCacheFile(new Path(queryFilePath).toUri(), conf);
        FileInputFormat.setInputPaths(conf, new Path(inputPath));
        FileOutputFormat.setOutputPath(conf, new Path(outputPath));

        JobClient.runJob(conf);
        return 0;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new Ngram(), args);
        System.exit(res);
    }
}
