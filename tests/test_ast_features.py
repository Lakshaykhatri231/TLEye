from tleye.features.ast_features import FEATURE_NAMES, extract

CPP_NESTED = b"""
#include <bits/stdc++.h>
int main(){ int n; std::cin>>n; std::vector<int> a(n);
  for(int i=0;i<n;i++) for(int j=0;j<n;j++){ if(a[i]<a[j]) std::swap(a[i],a[j]); }
  std::sort(a.begin(), a.end()); }
"""
PY_NESTED = b"""
def f(n):
    return 1 if n < 2 else f(n-1) + f(n-2)
n = int(input())
a = list(map(int, input().split()))
for i in range(n):
    for j in range(n):
        for k in range(n):
            pass
    a.sort()
b = [x for x in a for y in a]
"""
JAVA_SORT = b"""
import java.util.*;
public class Main { public static void main(String[] args){
  Scanner sc = new Scanner(System.in); int n = sc.nextInt(); int[] a = new int[n];
  for (int i=0;i<n;i++){ Arrays.sort(a); }
  while (true) { break; } } }
"""


def test_same_columns_all_languages():
    for code, lang in [(CPP_NESTED, "C++"), (PY_NESTED, "Python"), (JAVA_SORT, "Java")]:
        assert set(extract(code, lang)) == set(FEATURE_NAMES)


def test_cpp_nested_loops_and_sort():
    f = extract(CPP_NESTED, "C++")
    assert f["max_loop_depth"] == 2 and f["n_loops"] == 2 and f["n_loops_ge2"] == 1
    assert f["n_sort"] == 1 and f["sort_in_loop"] == 0
    assert f["recursive"] == 0 and f["io_slow_calls"] >= 1


def test_python_recursion_depth_and_comprehension():
    f = extract(PY_NESTED, "Python")
    assert f["recursive"] == 1 and f["n_rec_calls"] == 2
    assert f["max_loop_depth"] == 3          # the triple for-loop
    assert f["n_loops"] >= 5                 # 3 for + 2 comprehension clauses
    assert f["sort_in_loop"] == 1 and f["io_slow_calls"] == 2


def test_java_sort_in_loop_scanner_and_while_true():
    f = extract(JAVA_SORT, "Java")
    assert f["sort_in_loop"] == 1 and f["io_slow_calls"] == 1 and f["has_while_true"] == 1
    assert f["max_loop_depth"] == 1


def test_garbage_and_non_utf8_do_not_crash():
    assert extract(b"\xff\xfe not code (((", "C++")["n_chars"] > 0
    assert extract("# 日本語\nprint('こんにちは')".encode("shift_jis", "ignore"), "Python")["n_lines"] == 2
