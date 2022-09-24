# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     email_api
   Description:   ...
   Author:        cqh
   date:          2022/9/22 9:55
-------------------------------------------------
   Change Activity:
                  2022/9/22:
-------------------------------------------------
"""
__author__ = 'cqh'

import datetime
import sys
import time
import traceback
from email.mime.text import MIMEText
from email.header import Header
from smtplib import SMTP_SSL, SMTPException
from api_utils import covert_time_format

# sender_qq为发件人的qq号码
sender_qq = '1103878898'
# pwd为qq邮箱的授权码
pwd = 'baukkbwizzjwiidb'
# 收件人邮箱receiver
receiver = '1103878898@qq.com'
host_server = 'smtp.qq.com'
sender_qq_mail = sender_qq + '@qq.com'
time_format = "%Y-%m-%d %H:%M:%S"


def get_run_time(start, end):
    """
    字符串转时间戳
    :param start:
    :param end:
    :return:
    """
    start_time_date = time.strftime(time_format, time.localtime(start))
    end_time_date = time.strftime(time_format, time.localtime(end))

    return start_time_date, end_time_date, covert_time_format(end - start)


def send_an_error_message(program_name, error_name, error_detail):
    '''
    @program_name: 运行的程序名
    @error_name: 错误名
    @error_detail: 错误的详细信息
    @description: 程序出错是发送邮件提醒
    '''
    # 获取程序出错的时间
    error_time = datetime.datetime.strftime(datetime.datetime.today(), time_format)
    # 邮件内容
    subject = "【程序异常提醒】{name}-{date}".format(name=program_name, date=error_time)  # 邮件的标题
    content = '''<div class="emailcontent" style="width:100%;max-width:720px;text-align:left;margin:0 auto;padding-top:80px;padding-bottom:20px">
        <div class="emailtitle">
            <h1 style="color:#fff;background:#51a0e3;line-height:70px;font-size:24px;font-weight:400;padding-left:40px;margin:0">程序运行异常通知</h1>
            <div class="emailtext" style="background:#fff;padding:20px 32px 20px">
                <p style="color:#6e6e6e;font-size:13px;line-height:24px">程序：<span style="color:red;">【{program_name}】</span>运行过程中出现异常错误，下面是具体的异常信息，请及时核查处理！</p>
                <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border-top:1px solid #eee;border-left:1px solid #eee;color:#6e6e6e;font-size:16px;font-weight:normal">
                    <thead>
                        <tr>
                            <th colspan="2" style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;background:#f8f8f8">程序异常详细信息</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;width:100px">异常简述</td>
                            <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{error_name}</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center">异常详情</td>
                            <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{error_detail}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
        '''.format(program_name=program_name, error_name=error_name, error_detail=error_detail)  # 邮件的正文部分
    # 实例化一个文本对象
    massage = MIMEText(content, 'html', 'utf-8')
    massage['Subject'] = subject  # 标题
    massage['From'] = sender_qq_mail  # 发件人
    massage['To'] = receiver  # 收件人

    try:
        mail = SMTP_SSL(host_server, 465)  # 连接SMTP服务，默认465和944这里用994
        mail.login(sender_qq, pwd)  # 登录到SMTP服务
        mail.sendmail(sender_qq_mail, receiver, massage.as_string())  # 发送邮件
        print("成功发送了一封邮件到" + receiver)
    except SMTPException as ex:
        print("邮件发送失败！")
        print(ex)


def send_success_mail(program_name, run_start_time, run_end_time):
    """
    成功邮箱
    :param program_name:
    :param run_start_time: 时间戳
    :param run_end_time: 时间戳
    :return:
    """
    # 转成时间戳并计算
    start_date, end_date, run_time = get_run_time(run_start_time, run_end_time)
    # 获取程序的时间
    error_time = datetime.datetime.strftime(datetime.datetime.today(), time_format)
    # 邮件内容
    subject = "【程序运行完成提醒】{name}-{date}".format(name=program_name, date=error_time)  # 邮件的标题
    content = '''<div class="emailcontent" style="width:100%;max-width:720px;text-align:left;margin:0 auto;padding-top:80px;padding-bottom:20px">
            <div class="emailtitle">
                <h1 style="color:#fff;background:#51a0e3;line-height:70px;font-size:24px;font-weight:400;padding-left:40px;margin:0">程序运行完成通知</h1>
                <div class="emailtext" style="background:#fff;padding:20px 32px 20px">
                    <p style="color:#6e6e6e;font-size:13px;line-height:24px">程序：<span style="color:red;">【{program_file_name}】</span>运行成功!</p>
                    <table cellpadding="0" cellspacing="0" border="0" style="width:100%;border-top:1px solid #eee;border-left:1px solid #eee;color:#6e6e6e;font-size:16px;font-weight:normal">
                        <thead>
                            <tr>
                                <th colspan="2" style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;background:#f8f8f8">程序详细信息</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center;width:100px">程序开始运行时间</td>
                                <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{start_time}</td>
                            </tr>
                            <tr>
                                <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center">程序结束运行时间</td>
                                <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{end_time}</td>
                            </tr>
                            <tr>
                                <td style="padding:10px 0;border-right:1px solid #eee;border-bottom:1px solid #eee;text-align:center">程序总共运行时间</td>
                                <td style="padding:10px 20px 10px 30px;border-right:1px solid #eee;border-bottom:1px solid #eee;line-height:30px">{run_time}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
            '''.format(program_file_name=sys.argv, start_time=start_date, end_time=end_date,
                       run_time=run_time)  # 邮件的正文部分

    send_mail(subject, content)


def send_mail(mail_title='', mail_content=''):
    smtp = SMTP_SSL(host_server, 465)
    try:
        # ssl登录
        # set_debuglevel()是用来调试的。参数值为1表示开启调试模式，参数值为0关闭调试模式
        smtp.set_debuglevel(0)
        smtp.ehlo(host_server)
        smtp.login(sender_qq, pwd)

        msg = MIMEText(mail_content, "html", 'utf-8')
        msg["Subject"] = Header(mail_title, 'utf-8')
        msg["From"] = sender_qq_mail
        msg["To"] = receiver
        smtp.sendmail(sender_qq_mail, receiver, msg.as_string())
    except SMTPException as e:
        print('error', e)
    finally:
        smtp.quit()

    print("send email success!")


if __name__ == '__main__':
    # date_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # run_start_time = time.time()
    # time.sleep(1)
    #
    # date_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # run_end_time = time.time()
    # email_content = """
    # run_file: {},
    # run_start_time: {},
    # run_end_time: {},
    # run_time: {},
    # """.format(sys.argv, date_start_time, date_end_time, run_end_time - run_start_time)
    # # 邮件的正文内容
    # # mail_content = '你好，我陈钦海 ，现在在进行一项用python登录qq邮箱发邮件的测试'
    # # 邮件标题
    # mail_title = 'python {} file run success!'.format(sys.argv)
    # send_mail(mail_title=mail_title, mail_content=email_content)

    # try:
    #     b = int("测试")  # 执行程序位置
    # except Exception as ex:
    #     print(traceback.format_exc())
    #     send_an_error_message(program_name='程序测试', error_name=repr(ex), error_detail=traceback.format_exc())

    st = time.time()
    time.sleep(5)
    en = time.time()
    send_success_mail("test程序", st, en)
