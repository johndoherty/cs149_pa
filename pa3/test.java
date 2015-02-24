public class Test {
    public void readFile(Path file) {
        try {
            BufferedReader fis = new BufferedReader(new FileReader(file.toString()));
            String text = "";
            String line = null;
            while ((line = fis.readLine()) != null) {
                text += line;
            }
            System.out.println(text);
        } catch (IOException ioe) {
            System.err.println("Caught exception while parsing the cached file '" + patternsFile + "' : " + StringUtils.stringifyException(ioe));
        }
    }

    public static void main(String[] args) throws Exception {
        /*String pattern = "<title>(.+?)</title>";
        if (args[0].matches(pattern)) {
            String title = args[0].replaceAll(pattern, "$1");
            System.out.println("Title:" + title);
        } else {
            System.out.println("No match");
        }*/
        readFile(args[0]);
        return;
    }
}
