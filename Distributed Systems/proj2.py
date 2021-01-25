import socket
import pickle
from threading import Thread
import threading
import sys
import copy
import operator
from random import randint
from time import sleep

class Event:
    def __init__(self, name, day, start, end, participants, event_type):
        self.name = name
        self.day = day
        self.start = start
        self.end = end
        self.participants = participants
        self.event_type = event_type
    def __eq__(self, other):
        if isinstance(other, Event):
            return self.name == other.name
        return False
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        
def save_to_file(filename, data):
    with open(filename, "wb+") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def tcp_send(data, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        pickled_data = pickle.dumps(data)
        length_str = str(len(pickled_data)) + "\n"
        s.send(length_str.encode())
        s.send(pickled_data)
    except socket.error:
        pass
    
def tcp_receive():
    for i in range(2):
        try:
            (client_socket, address) = main_s.accept()
            try:
                buf = ''
                while "\n" not in buf:
                    buf += client_socket.recv(1).decode()
                msg_size = int(buf)
                msg = client_socket.recv(msg_size)
                return pickle.loads(msg)
            except socket.error:
                return ("recv_error", "recv_error", "recv_error", "recv_error", "recv_error")
        except socket.error:
            sleep(randint(1,5)/10.0)
    return ("timeout", "timeout", "timeout", "timeout", "timeout")
        
def broadcast_event(slot, msg_type, m, v, hostname):
    for host in host_list:
        if host != my_hostname:
            tcp_send((slot, msg_type, m, v, hostname), host, host_list[host])
            
def find_conflicts(event):
    conflicts = []
    for event2 in calendar:
        if event.day == event2.day and event.start < event2.end and event2.start < event.end and my_hostname in event2.participants:
            conflicts.append(event2)
    conflicts.sort(key=operator.attrgetter('name'))
    return conflicts
    
def read_terminal_input():
    global my_log
    global my_hostname
    global my_port
    global host_list
    global calendar
    while True:
        input_str = input(">> ")
        input_arr = input_str.split(" ")
        if input_arr[0] == "schedule":
            schedule(input_arr)
        elif input_arr[0] == "cancel":
            cancel(input_arr)
        elif input_arr[0] == "view":
            view()
        elif input_arr[0] == "myview":
            myview()
        elif input_arr[0] == "log":
            log()
        elif input_arr[0] == "exit":
            exit()
            return
        else:
            print("Error: unknown input")
            
def exit():
    lock.acquire()
    exit = True
    lock.release()
            
def schedule(arr):
    # Schedule meeting
    lock.acquire()
    event = Event(arr[1], arr[2], arr[3], arr[4], arr[5].split(','), "schedule")
    if not find_conflicts(event):
        prepare(max(my_log.keys(), default=-1) + 1, event, 0)
    else:
        print("Unable to schedule meeting {}.".format(event.name))
    lock.release()
    
def cancel(arr):
    # Cancel meeting
    lock.acquire()

    # Look for meeting
    event = None
    for event2 in calendar:
        if event2.name == arr[1]:
            event = copy.deepcopy(event2)
    if event is None:
        print("Unable to cancel {}.".format(arr[1]))
    else:
        event.event_type = "cancel"
        prepare(max(my_log.keys(), default=0) + 1, event, 0)
    lock.release()
    
def view():
    lock.acquire()
    view_calendar = copy.deepcopy(calendar)
    view_calendar.sort(key=operator.attrgetter('day', 'start', 'name'))
    for event in view_calendar:
        print("{} {} {} {} {}".format(event.name, event.day, event.start, event.end, ",".join(event.participants)))
    lock.release()
    
def myview():
    lock.acquire()
    view_calendar = copy.deepcopy(calendar)
    view_calendar.sort(key=operator.attrgetter('day', 'start', 'name'))
    for event in view_calendar:
        if my_hostname in event.participants:
            print("{} {} {} {} {}".format(event.name, event.day, event.start, event.end, ",".join(event.participants)))
    lock.release()

def log():
    lock.acquire()
    for i in sorted(my_log.keys()):
        print("{} {} {} {} {} {}".format(my_log[i].event_type, my_log[i].name, my_log[i].day, my_log[i].start, my_log[i].end, ",".join(my_log[i].participants)))
    lock.release()

def prepare(slot, event, counter):
    m = int(str(counter) + str(site_id))
    broadcast_event(slot, "prepare", m, None, my_hostname)
    proposer_dict[slot] = [0, event, 0, m]

def promise(accNum, accVal, slot):
    if proposer_dict[slot][0] < accNum:
        proposer_dict[slot][0] = accNum
        proposer_dict[slot][1] = accVal
    proposer_dict[slot][2] += 1
    if proposer_dict[slot][2] > (len(host_list)-1)/2.0:
        broadcast_event(slot, "accept", proposer_dict[slot][3], proposer_dict[slot][1], my_hostname)
        proposer_dict[slot] = [0, proposer_dict[slot][1], 0, proposer_dict[slot][3]]

def accept(m, v, slot, hostname):
    if m >= acceptor_dict[slot][2]:
        acceptor_dict[slot] = [m, v, m]
        tcp_send((slot, "ack", m, v, my_hostname), hostname, host_list[hostname])

def ack(accNum, accVal, slot):
    proposer_dict[slot][2] += 1
    if proposer_dict[slot][2] > (len(host_list)-1)/2.0:
        broadcast_event(slot, "commit", proposer_dict[slot][3], proposer_dict[slot][1], my_hostname)
        commit(proposer_dict[slot][1], slot)

def commit(v, slot):
    my_log[slot] = v
    if slot in proposer_dict.keys() and proposer_dict[slot][1] == v:
        if v.event_type == "schedule":
            print("Meeting {} scheduled".format(v.name))
        else:
            print("Meeting {} cancelled".format(v.name))
    if slot in proposer_dict.keys():
        del proposer_dict[slot]
    if v.event_type == "schedule":
        calendar.append(v)
        for i in range(slot+1, max(my_log.keys())):
            if my_log[slot].name == my_log[i] and my_log[slot].event_type != my_log[i].event_type:
                calendar.remove(v)
                break
    else:
        try:
            calendar.remove(v)
        except:
            pass
    missing_slot_check(my_log)
    if len(my_log.keys())%5 == 0:
        checkpoint()

def build_calendar(my_log, acc_dict):
    calendar = []
    for key in sorted(my_log.keys()):
        if my_log[key].event_type == "schedule":
            calendar.append(my_log[key])
        elif my_log[key].event_type == "cancel":
            try:
                calendar.remove(my_log[key])
            except:
                continue
    return calendar
    
def checkpoint():
    save_to_file("my_log_{}.pickle".format(my_hostname), my_log)
    save_to_file("acceptor_dict_{}.pickle".format(my_hostname), acceptor_dict)
    save_to_file("proposer_dict_{}.pickle".format(my_hostname), proposer_dict)

def missing_slot_check(log):
    for i in range(max(log.keys())):
        if i not in log.keys():
            prepare(i, None, 0)

def read_tcp():
    missing_slot_check(my_log)
    while True:
        (tcp_slot, tcp_msg_type, tcp_m, tcp_v, tcp_host) = tcp_receive()
        lock.acquire()
    
        if exit:
            lock.release()
            break
        
        if tcp_slot == "recv_error":
            pass
        elif tcp_slot == "timeout":
            for slot in proposer_dict:
                counter = proposer_dict[slot][3]//10
                prepare(slot, proposer_dict[slot][1], counter+1)
        else:
            if tcp_msg_type == "prepare":
                if tcp_slot not in acceptor_dict.keys() or tcp_m > acceptor_dict[tcp_slot][2]:
                    if tcp_slot not in acceptor_dict.keys():
                        tcp_send((tcp_slot, "promise", 0, None, my_hostname), tcp_host, host_list[tcp_host])
                        acceptor_dict[tcp_slot] = [0, None, tcp_m]
                    else:
                        tcp_send((tcp_slot, "promise", acceptor_dict[tcp_slot][0], acceptor_dict[tcp_slot][1], my_hostname), tcp_host, host_list[tcp_host])
                        acceptor_dict[tcp_slot][2] = tcp_m
            elif tcp_msg_type == "promise":
                promise(tcp_m, tcp_v, tcp_slot)
            elif tcp_msg_type == "accept":
                accept(tcp_m, tcp_v, tcp_slot, tcp_host)
            elif tcp_msg_type == "ack":
                ack(tcp_m, tcp_v, tcp_slot)
            elif tcp_msg_type == "commit":
                commit(tcp_v, tcp_slot)
        
        lock.release()

if __name__ == '__main__':
    # Read in host names and ports
    site_id = 0
    i = 1
    my_hostname = sys.argv[1]
    
    host_list = {}
    host_file = open("./knownhosts_tcp.txt")
    for line in host_file:
        new_host = line.split(' ')
        host_list[new_host[0]] = int(new_host[1].strip())
        if new_host[0] == my_hostname:
            site_id = i
        i += 1

    # Determine my hostname and port
    my_port = None
    for host in host_list:
        if host == my_hostname:
            my_port = host_list[host]
    process_num = len(host_list)

    # Stored data
    my_log = {}
    try:
        with open('partial_my_log_{}.pickle'.format(my_hostname), 'rb') as handle:
            my_log = pickle.load(handle)
    except OSError:
        save_to_file("partial_my_log_{}.pickle".format(my_hostname), my_log)
    #Per log entry, store maxAccNum, v, numAcc, n
    proposer_dict = {}
    #Per log entry, stores accNum, accVal, maxPrepare
    acceptor_dict = {}

    calendar = build_calendar(my_log, acceptor_dict)
    
    exit = False
        
    main_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    main_s.settimeout(0.5)
    lock = threading.Lock()

    # Initialize socket
    main_s.bind((my_hostname, my_port))
    main_s.listen(process_num - 1)

    # Run program
    thread = Thread(target=read_tcp)
    thread.start()
    read_terminal_input()