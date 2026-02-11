package org.InvertedIndex;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.LineReader;

import java.io.*;
import java.net.URI;
import java.util.*;
import java.text.Normalizer;

public class InvertedIndex {

    public static class WordFileWritable implements WritableComparable<WordFileWritable> {
        private Text word;
        private Text fileName;

        public WordFileWritable() {
            this.word = new Text();
            this.fileName = new Text();
        }

        public WordFileWritable(String word, String fileName) {
            this.word = new Text(word);
            this.fileName = new Text(fileName);
        }

        public void write(DataOutput out) throws IOException {
            word.write(out);
            fileName.write(out);
        }

        public void readFields(DataInput in) throws IOException {
            word.readFields(in);
            fileName.readFields(in);
        }

        public int compareTo(WordFileWritable o) {
            int cmp = word.compareTo(o.word);
            if (cmp != 0) return cmp;
            return fileName.compareTo(o.fileName);
        }

        public Text getWord() { return word; }
        public Text getFileName() { return fileName; }

        @Override
        public String toString() {
            return word + "@" + fileName;
        }
    }

    public static class WordGroupingComparator extends WritableComparator {
        protected WordGroupingComparator() {
            super(WordFileWritable.class, true);
        }

        public int compare(WritableComparable w1, WritableComparable w2) {
            WordFileWritable a = (WordFileWritable) w1;
            WordFileWritable b = (WordFileWritable) w2;
            return a.getWord().compareTo(b.getWord());
        }
    }

    public static class InverseIndexMapper extends Mapper<LongWritable, Text, WordFileWritable, LongWritable> {
        private Set<String> stopwords = new HashSet<>();
        private String fileName;

        @Override
        protected void setup(Context context) throws IOException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles != null) {
                for (URI cacheFile : cacheFiles) {
                    Path path = new Path(cacheFile.getPath());
                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(FileSystem.get(context.getConfiguration()).open(path)))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            stopwords.add(line.trim().toLowerCase());
                        }
                    }
                }
            }
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            fileName = fileSplit.getPath().getName();
        }

        public static String removeAccents(String input) {
            String normalized = Normalizer.normalize(input, Normalizer.Form.NFD);
            return normalized.replaceAll("\\p{M}", "");
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            Set<String> uniqueWords = new HashSet<>();
            String normalized = removeAccents(Normalizer.normalize(value.toString(), Normalizer.Form.NFKC));
            String cleaned = normalized
                    .toLowerCase()
                    .replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]+", " ")
                    .replaceAll("\\s+", " ")
                    .trim();

            for (String word : cleaned.split(" ")) {
                if (!word.isEmpty() && !stopwords.contains(word)) {
                    uniqueWords.add(word);
                }
            }

            for (String word : uniqueWords) {
                context.write(new WordFileWritable(word, fileName), key);
            }
        }
    }

    public static class InverseIndexReducer extends Reducer<WordFileWritable, LongWritable, Text, Text> {
        @Override
        protected void reduce(WordFileWritable key, Iterable<LongWritable> values, Context context)
                throws IOException, InterruptedException {

            Map<String, Set<Long>> fileToLines = new HashMap<>();

            for (LongWritable val : values) {
                String file = key.getFileName().toString();
                fileToLines.putIfAbsent(file, new TreeSet<>());
                fileToLines.get(file).add(val.get());
            }

            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, Set<Long>> entry : fileToLines.entrySet()) {
                if (sb.length() > 0) sb.append(" ");
                sb.append("(").append(entry.getKey()).append(", ");
                sb.append(String.join(", ",
                        entry.getValue().stream().map(String::valueOf).toArray(String[]::new)));
                sb.append(")");
            }

            context.write(new Text(key.getWord()), new Text(sb.toString()));
        }
    }


    public static class GlobalLineInputFormat extends FileInputFormat<LongWritable, Text> {
        @Override
        public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context) {
            return new GlobalLineRecordReader();
        }
    }

    public static class GlobalLineRecordReader extends RecordReader<LongWritable, Text> {
        private LineReader in;
        private LongWritable key = new LongWritable();
        private Text value = new Text();
        private long currentLineNumber = 0;

        @Override
        public void initialize(InputSplit genericSplit, TaskAttemptContext context) throws IOException {
            FileSplit split = (FileSplit) genericSplit;
            Configuration job = context.getConfiguration();
            Path file = split.getPath();
            FileSystem fs = file.getFileSystem(job);
            FSDataInputStream fileIn = fs.open(file);
            in = new LineReader(fileIn, job);

            long splitStart = split.getStart();
            long currentPos = 0;
            Text dummy = new Text();

            while (currentPos < splitStart) {
                int bytesRead = in.readLine(dummy);
                if (bytesRead == 0) break;
                currentPos += bytesRead;
                currentLineNumber++;
            }

            currentLineNumber++; 
        }

        @Override
        public boolean nextKeyValue() throws IOException {
            int lineSize = in.readLine(value);
            if (lineSize == 0) return false;
            key.set(currentLineNumber++);
            return true;
        }

        @Override public LongWritable getCurrentKey() { return key; }
        @Override public Text getCurrentValue() { return value; }
        @Override public float getProgress() { return 0; }
        @Override public void close() throws IOException { in.close(); }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: InvertedIndex <input path> <output path>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        conf.set("mapreduce.input.fileinputformat.split.maxsize", "524288");
        conf.set("mapreduce.input.fileinputformat.split.minsize", "524288");

        Job job = Job.getInstance(conf, "Inverted Index with Global Line Numbers");
        job.setJarByClass(InvertedIndex.class);

        job.setMapperClass(InverseIndexMapper.class);
        job.setReducerClass(InverseIndexReducer.class);

        job.setMapOutputKeyClass(WordFileWritable.class);
        job.setMapOutputValueClass(LongWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setInputFormatClass(GlobalLineInputFormat.class);
        job.setGroupingComparatorClass(WordGroupingComparator.class);

        job.addCacheFile(new URI("/stopwords/stopwords.txt"));

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
