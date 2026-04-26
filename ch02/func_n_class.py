# 함수로 덧셈기를 만들면 어떤 식이든 이런 전역 변수를 쓰기 쉽다...
result1 = 0
result2 = 0


# 그리고 독립적으로 실행되는 두 개의 덩어리를 만들고 싶으면 선언부에서 두 번 선언해야 한다...
def add1(num):
    # 전역 변수를 함수 내에서 사용하려면 global
    global result1
    result1 += num
    return result1


def add2(num):
    global result2
    result2 += num
    return result2


# 같은 효과를 클래스를 통해서 만들면 한 번의 선언과 인스턴스 변수로 처리가 가능해진다...
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result


if __name__ == "__main__":
    print("첫 번째 계산기 누적")
    print(add1(3))
    print(add1(4))
    print("두 번째 계산기 누적")
    print(add2(3))
    print(add2(7))

    print("첫 번째 계산기 클래스 누적")
    cal1 = Calculator()
    print(cal1.add(3))
    print(cal1.add(4))
    print("두 번째 계산기 클래스 누적")
    cal2 = Calculator()
    print(cal2.add(3))
    print(cal2.add(7))
