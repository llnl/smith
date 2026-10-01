#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET


def test_times(path):
    return {
        test.attrib["name"]: float(test.attrib.get("time", 0.0))
        for test in ET.parse(path).iter("testcase")
    }


if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} BEFORE_JUNIT_XML AFTER_JUNIT_XML")

before = test_times(sys.argv[1])
after = test_times(sys.argv[2])
common = before.keys() & after.keys()

print(f"{'test':50} {'before':>9} {'after':>9} {'saved':>9}")
for name in sorted(common, key=lambda name: before[name] - after[name], reverse=True):
    print(f"{name:50} {before[name]:9.2f} {after[name]:9.2f} {before[name] - after[name]:+9.2f}")

before_total = sum(before[name] for name in common)
after_total = sum(after[name] for name in common)
percent = 100.0 * (before_total - after_total) / before_total if before_total else 0.0
print(f"\n{len(common)} common tests: {before_total:.2f}s -> {after_total:.2f}s "
      f"({before_total - after_total:+.2f}s, {percent:+.1f}%)")
