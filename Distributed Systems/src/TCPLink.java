import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.net.InetAddress;
import java.net.Socket;

public class TCPLink implements Link {

    private Socket socket;
    private ObjectOutputStream outputStream;
    private ObjectInputStream inputStream;

    public void open(String host, int port){
        try {
            socket = new Socket(host, port);
            socket.bind();
            outputStream = new ObjectOutputStream(socket.getOutputStream());
            inputStream = new ObjectInputStream(socket.getInputStream());
        } catch (IOException ie) {
            ie.printStackTrace();
        }
    }

    public void close() {
        try {
            socket.close();
        } catch (IOException ie) {
            ie.printStackTrace();
        }

    }

    public void send(Serializable object) {
        try {
            outputStream.writeObject(object);
            outputStream.flush();
        } catch (IOException ie) {
            ie.printStackTrace();
        }
    }

    public Object receive() {
        try {
            return inputStream.readObject();
        } catch (IOException ie) {
            ie.printStackTrace();
        } catch (ClassNotFoundException ce) {
            ce.printStackTrace();
        }
        return null;
    }

}
