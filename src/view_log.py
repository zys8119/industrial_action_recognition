#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志查看和分析工具
"""

import argparse
from pathlib import Path
from datetime import datetime
import re

def parse_log_line(line):
    """解析日志行"""
    # 格式: [2026-03-10 11:08:30.123] 消息内容
    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] (.+)'
    match = re.match(pattern, line)
    if match:
        timestamp_str, message = match.groups()
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
        return timestamp, message
    return None, line

def analyze_log(log_file):
    """分析日志文件"""
    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"📊 分析日志: {log_file}\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 统计信息
    total_lines = len(lines)
    action_detections = []
    sessions = []
    current_session = None
    
    for line in lines:
        timestamp, message = parse_log_line(line)
        
        if '会话开始' in message:
            current_session = {'start': timestamp, 'detections': 0}
        elif '会话结束' in message:
            if current_session:
                current_session['end'] = timestamp
                sessions.append(current_session)
        elif '检测到动作' in message:
            if current_session:
                current_session['detections'] += 1
            
            # 解析动作信息
            # 格式: 检测到动作: welding | 置信度: 85% | 位置: (100,150)-(300,400) | 尺寸: 200x250
            parts = message.split('|')
            if len(parts) >= 4:
                action = parts[0].split(':')[1].strip()
                confidence = parts[1].split(':')[1].strip()
                position = parts[2].split(':')[1].strip()
                size = parts[3].split(':')[1].strip()
                
                action_detections.append({
                    'timestamp': timestamp,
                    'action': action,
                    'confidence': confidence,
                    'position': position,
                    'size': size
                })
    
    # 输出统计
    print(f"📝 总日志行数: {total_lines}")
    print(f"🎬 会话数量: {len(sessions)}")
    print(f"🎯 动作检测次数: {len(action_detections)}\n")
    
    # 会话统计
    if sessions:
        print("=" * 60)
        print("会话统计:")
        print("=" * 60)
        for i, session in enumerate(sessions, 1):
            duration = (session['end'] - session['start']).total_seconds()
            print(f"\n会话 {i}:")
            print(f"  开始时间: {session['start'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  结束时间: {session['end'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  持续时间: {duration:.1f} 秒")
            print(f"  检测次数: {session['detections']}")
    
    # 动作统计
    if action_detections:
        print("\n" + "=" * 60)
        print("动作类别统计:")
        print("=" * 60)
        
        action_counts = {}
        for det in action_detections:
            action = det['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(action_detections) * 100
            print(f"  {action}: {count} 次 ({percentage:.1f}%)")
        
        # 最近的检测
        print("\n" + "=" * 60)
        print("最近 10 次检测:")
        print("=" * 60)
        for det in action_detections[-10:]:
            time_str = det['timestamp'].strftime('%H:%M:%S')
            print(f"  [{time_str}] {det['action']} | {det['confidence']} | "
                  f"位置: {det['position']} | 尺寸: {det['size']}")

def tail_log(log_file, lines=20):
    """显示日志尾部"""
    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    print(f"📄 最后 {lines} 行日志:\n")
    for line in all_lines[-lines:]:
        print(line.rstrip())

def watch_log(log_file):
    """实时监控日志"""
    import time
    
    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"👀 实时监控日志: {log_file}")
    print("按 Ctrl+C 退出\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        # 移动到文件末尾
        f.seek(0, 2)
        
        try:
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 停止监控")

def main():
    parser = argparse.ArgumentParser(description='日志查看和分析工具')
    parser.add_argument('--log', type=str, default='logs/recognition.log',
                       help='日志文件路径')
    parser.add_argument('--analyze', action='store_true',
                       help='分析日志统计信息')
    parser.add_argument('--tail', type=int, metavar='N',
                       help='显示最后 N 行')
    parser.add_argument('--watch', action='store_true',
                       help='实时监控日志')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_log(args.log)
    elif args.tail:
        tail_log(args.log, args.tail)
    elif args.watch:
        watch_log(args.log)
    else:
        # 默认显示最后 20 行
        tail_log(args.log, 20)

if __name__ == "__main__":
    main()
