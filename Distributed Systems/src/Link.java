import java.io.Serializable;

public interface Link {
    void open(String host, int port);
    void close();
    void send(Serializable object);
    Object receive();
}
