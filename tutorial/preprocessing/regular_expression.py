import re
'''
. 기호
'''
r1 = re.compile("a.c")
print(r1.search("kkk"))
print(r1.search("abc"))
'''
? 기호 : 있을수도 있고 없을수도 있다.
'''
r2 = re.compile("ab?c")
print(r2.search("abbc"))
print(r2.search("abc"))
print(r2.search("ac"))
'''
* 기호 : 개수가 0 이상
'''
r3 = re.compile("ab*c")
print(r3.search("a"))
print(r3.search("abc"))
print(r3.search("abbbbc"))
'''
+기호 : 개수가 1개 이상
'''
r4=re.compile("ab+c")
print(r4.search("ac"))
print(r4.search("abc"))
print(r4.search("abbbbbc"))
'''
^ 기호 : 시작 글자 지정
'''
r5=re.compile("^a")
print(r5.search("bbc"))
print(r5.search("ab"))



