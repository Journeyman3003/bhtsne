#!/usr/bin/env python

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


SMTP_SERVER = 'smtp.gmail.com'
PORT = 465  # For SSL
SENDER_EMAIL = "tsne.status@gmail.com"  # Enter your address
RECEIVER_EMAIL = "toby.mai@web.de"  # Enter receiver address
PASSWORD = "tdstrbtdstchstcnghbrmbddng"


def send_error(logfile_name, logfile_path, tsne_main_call):
    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = SENDER_EMAIL

    # storing the receivers email address
    msg['To'] = RECEIVER_EMAIL

    # storing the subject
    msg['Subject'] = "t-SNE crashed"

    # string to store the body of the mail

    body = "Execution crashed: {}\nWhoops! The process died, please refer to logfile.".format(" ".join(tsne_main_call))

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # attach logfile
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')
    # To change the payload into encoded form
    p.set_payload((open(logfile_path, "rb")).read())
    # encode into base64
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % logfile_name)
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # Converts the Multipart msg into a string
    text = msg.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, PORT, context=context) as server:
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)


def send_mail(logfile_name, logfile_path, result_archive_name, result_archive_path, tsne_main_call):
    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = SENDER_EMAIL

    # storing the receivers email address
    msg['To'] = RECEIVER_EMAIL

    # storing the subject
    msg['Subject'] = "t-SNE finished"

    # string to store the body of the mail
    body = "Execution finished: {}\nThis message is sent from Python.".format(" ".join(tsne_main_call))

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # attach logfile
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')
    # To change the payload into encoded form
    p.set_payload((open(logfile_path, "rb")).read())
    # encode into base64
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % logfile_name)
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # attach result zip
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'zip')
    # To change the payload into encoded form
    p.set_payload((open(result_archive_path, "rb")).read())
    # encode into base64
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % result_archive_name)
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # Converts the Multipart msg into a string
    text = msg.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, PORT, context=context) as server:
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)


if __name__ == '__main__':
    send_mail("","","","","")
