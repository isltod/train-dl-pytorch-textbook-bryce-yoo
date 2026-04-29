import re

# . 임의 문자열 하나
r = re.compile("a.c")
# 매치가 안되면 아무 결과도 없는 것이 아니라 None 반환
print(r.search("kkk"))
print(r.search("abc"))

# ? 바로 앞의 문자열 1개 있거나 없거나...
r = re.compile("ab?c")
print(r.search("abbc"))
print(r.search("abc"))
print(r.search("ac"))

# * 바로 앞의 문자열 몇개든 있거나 없거나...
r = re.compile("ab*c")
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbc"))

# + 바로 앞의 문자열 몇개든 있음, 없으면 안된다...
r = re.compile("ab+c")
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbc"))

# ^ 뒤 문자열들로 시작되야 한다...
r = re.compile("^ab")
print(r.search("bbc"))
print(r.search("zac"))
print(r.search("abz"))

# {숫자} 바로 앞 문자열이 몇 번 반복되나
r = re.compile("ab{2}c")
print(r.search("abc"))
print(r.search("abbc"))

# {숫자1, 숫자2} 바로 앞 문자열이 숫자1~숫자2 사이 반복
r = re.compile("ab{2,8}")
print(r.search("abc"))
print(r.search("abbbbc"))

#  [] 대괄호 안의 문자들 중 하나, 여럿이면 처음 매치
r = re.compile("[abc]")
print(r.search("efz"))
print(r.search("a"))
print(r.search("bc"))

# [문자1-문자2] 범위로 지정
r = re.compile("[a-z]")
print(r.search("A3B"))
print(r.search("aBC"))

# [^문자열] 문자열은 제외
r = re.compile("[^abc]")
print(r.search("bc"))
print(r.search("bd"))

# match는 맨 앞부터 맞아야만, search는 전체에서
r = re.compile("ab.")
print(r.match("kkkabc"))
print(r.match("kabckk"))
print(r.match("abckkk"))
print(r.search("kkkabc"))

# findall 모두 찾아서 리스트로
text = """이름: 김철수
전화번호: 010-1234-5678
나이: 30
성별: 남
"""
# 숫자 하나 이상
print(re.findall("\d+", text))
