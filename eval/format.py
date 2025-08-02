# -----------------------------------------------------------------------
# BITS official code : eval/format.py
# -----------------------------------------------------------------------
# Modified from EDD (https://github.com/ibm-aur-nlp/EDD)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

# Helper function to read in tables from the annotations
# from bs4 import BeautifulSoup as bs
from html import escape


def format_html(img, wofunc=False):
    """
    Formats HTML code from tokenized annotation of img
    """
    html_code = img['structure']['tokens'].copy()
    if len(html_code) > 0 and html_code[0] == '<table>' and html_code[-1] == '</table>':
        html_code = html_code[1:-1]
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    if wofunc:
        if '<thead>' in html_code:
            html_code.remove('<thead>')
        if '</thead>' in html_code:
            html_code.remove('</thead>')
        if '<tbody>' in html_code:
            html_code.remove('<tbody>')
        if '</tbody>' in html_code:
            html_code.remove('</tbody>')
    html_code = ''.join(html_code)
    html_code = '''<html><body><table>%s</table></body></html>''' % html_code
    # html_code = '''<html>
    #                <head>
    #                <meta charset="UTF-8">
    #                <style>
    #                table, th, td {
    #                  border: 1px solid black;
    #                  font-size: 10px;
    #                }
    #                </style>
    #                </head>
    #                <body>
    #                <table frame="hsides" rules="groups" width="100%%">
    #                  %s
    #                </table>
    #                </body>
    #                </html>''' % html_code
    #
    # # prettify the html
    # soup = bs(html_code)
    # html_code = soup.prettify()
    return html_code