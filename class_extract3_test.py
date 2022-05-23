import paramiko
import sys
import os

def main():


    if __name__ == '__main__':
        try:
            if sys.argv[1] == 'deploy':
                # Connect to remote host
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect('ribble.cs.ucl.ac.uk', username='yyu', password='QnfJNu4rEV.Nya@bXUuv')

                # Setup sftp connection and transmit this script
                sftp = client.open_sftp()
                sftp.put(__file__, 'confidence_css/class_extract3.py')
                sftp.put(__file__, 'confidence_css/containers.py')
                sftp.close()

                # Run the transmitted script remotely without args and show its output.
                # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
                stdout = client.exec_command('python confidence_css/class_extract3.py')[1]

        except IndexError:
            pass

# No cmd-line args provided, run script normally
main()


