#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

def read_scp(scp_file):
    key_value_list = []
    with open(scp_file,"r") as fp:
        line = fp.readline()
        while line:
            tokens = line.strip().split()
            key = tokens[0]
            value = " ".join(tokens[1:])
            key_value_list.append((key, value))
            line = fp.readline()
    return key_value_list
