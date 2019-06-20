#!/usr/bin/env python

import smtplib, ssl


smtp_server = 'smtp.gmail.com'

port = 465  # For SSL
sender_email = "tsne.status@gmail.com"  # Enter your address
receiver_email = "toby.mai@web.de"  # Enter receiver address
password = "tdstrbtdstchstcnghbrmbddng"
message = """\
Subject: t-SNE finished

This message is sent from Python."""


def send_mail():
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


if __name__ == '__main__':
    send_mail()
