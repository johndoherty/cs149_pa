// ChatServer

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.Thread;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ChatServer {
    private static final Charset utf8 = Charset.forName("UTF-8");

    private static final String OK = "200 OK";
    private static final String NOT_FOUND = "404 NOT FOUND";
    private static final String HTML = "text/html";
    private static final String TEXT = "text/plain";
    private static final String DEFAULT_ROOM = "DEFAULT";

    private static final int NUM_CONSUMERS = 8;

    private static final Pattern PAGE_REQUEST
        = Pattern.compile("GET /([\\p{Alnum}]*/?) HTTP.*");
    private static final Pattern PULL_REQUEST
        = Pattern.compile("POST /([\\p{Alnum}]*)/?pull\\?last=([0-9]+) HTTP.*");
    private static final Pattern PUSH_REQUEST
        = Pattern.compile("POST /([\\p{Alnum}]*)/?push\\?msg=([^ ]*) HTTP.*");

    private static final String CHAT_HTML;
    static {
        try {
            CHAT_HTML = getFileAsString("../index.html");
        } catch (final IOException xx) {
            throw new Error("unable to start server", xx);
        }
    }

    private final Queue<Socket> connections = new LinkedList<Socket>();

    private final Set<Consumer> consumers = new HashSet<Consumer>();
    private final int port;
    private final Map<String,ChatState> stateByName
        = new HashMap<String,ChatState>();

    private class Consumer extends Thread {
        public void run() {
            while (true) {
                final Socket connection;
                synchronized(connections) {
                    while (connections.isEmpty()) {
                        try {
                            connections.wait();
                        } catch (InterruptedException e) {
                            System.err.println("InterruptedException: " + e.getMessage());
                        }
                    } 
                    connection = connections.remove();
                }

                try {
                    ChatServer.this.handle(connection);
                } catch (IOException e) {
                    System.err.println("IOException: " + e.getMessage());
                }
            }
        } 
    }

    /**
     * Constructs a new {@link ChatServer} that will service requests
     * on the specified <code>port</code>. <code>state</code> will be
     * used to hold the current state of the chat.
     */
    public ChatServer(final int port) throws IOException {
        this.port = port;
    }

    public void runForever() throws IOException {
        @SuppressWarnings("resource")
            final ServerSocket server = new ServerSocket(port);

        for (int i = 0; i < NUM_CONSUMERS; i++) {
            Consumer c = new Consumer();
            c.start();
            consumers.add(c);
        }
        while (true) {
            final Socket connection = server.accept();
            synchronized(connections) {
                connections.add(connection);
                connections.notifyAll();
            }
            for (Consumer c : consumers) {
                //System.out.println(c + ": " + c.status);
            }
        }
    }
    
    private static String replaceEmptyWithDefaultRoom(final String room) {
    	if (room.isEmpty()) {
    		return DEFAULT_ROOM;
    	}
    	return room;
    }

    private void handle(final Socket connection) throws IOException {
        try {
            final BufferedReader xi
                = new BufferedReader(new InputStreamReader(connection.getInputStream()));
            final OutputStream xo = new BufferedOutputStream(connection.getOutputStream());

            final String request = xi.readLine();
            System.out.println(Thread.currentThread() + ": " + request);

            Matcher m;
            if (request == null) {
                sendResponse(xo, NOT_FOUND, TEXT, "Empty request.");
            } else if (PAGE_REQUEST.matcher(request).matches()) {
                sendResponse(xo, OK, HTML, CHAT_HTML);
            } else if ((m = PULL_REQUEST.matcher(request)).matches()) {
                String room = replaceEmptyWithDefaultRoom(m.group(1));
                final long last = Long.valueOf(m.group(2));
                sendResponse(xo, OK, TEXT, getState(room).recentMessages(last));
            } else if ((m = PUSH_REQUEST.matcher(request)).matches()) {
                String room = replaceEmptyWithDefaultRoom(m.group(1));
                final String msg = m.group(2);
                getState(room).addMessage(msg);
                sendResponse(xo, OK, TEXT, "ack");
            } else {
                sendResponse(xo, NOT_FOUND, TEXT, "Malformed request.");
            }
        } finally {
        	connection.close();
        }
    }
    
    /**
     * Writes a minimal but valid HTTP response to
     * <code>output</code>.
     */
    private static void sendResponse(final OutputStream xo,
                                     final String status,
                                     final String contentType,
                                     final String content) throws IOException {
        final byte[] data = content.getBytes(utf8);
        final String headers =
            "HTTP/1.0 " + status + "\r\n" +
            "Content-Type: " + contentType + "; charset=utf-8\r\n" +
            "Content-Length: " + data.length + "\r\n\r\n";

        xo.write(headers.getBytes(utf8));
        xo.write(data);
        xo.flush();

        System.out.println(Thread.currentThread() + ": replied with " + data.length + " bytes");
    }

    private ChatState getState(final String room) {
        ChatState state = null;
        synchronized(stateByName) {
            state = stateByName.get(room);
            if (state == null) {
                state = new ChatState(room);
                stateByName.put(room, state);
            }

        }
        return state;
    }

    /**
     * Reads the resource with the specified path as a string, and
     * then returns the string.
     */
    private static String getFileAsString(final String path)
        throws IOException {
            byte[] fileBytes = Files.readAllBytes(Paths.get(path));
            return new String(fileBytes, utf8);
        }

    /**
     * Runs a chat server, with a default port of 8080.
     */
    public static void main(final String[] args) throws IOException {
        final int port = args.length == 0 ? 8080 : Integer.parseInt(args[0]);
        new ChatServer(port).runForever();
    }
}
