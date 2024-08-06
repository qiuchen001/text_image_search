import socket


class NetHelper:
    def __init__(self):
        self.name = 'net-helper'

    @classmethod
    def get_host_ip(cls):
        st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            st.connect(('10.255.255.255', 1))
            IP = st.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
            # IP = '10.66.8.51'
        finally:
            st.close()
        return IP


net_helper = NetHelper()

if __name__ == '__main__':
    print(net_helper.get_host_ip())
