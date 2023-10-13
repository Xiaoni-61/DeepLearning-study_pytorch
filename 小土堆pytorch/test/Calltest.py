class Person:
    def __call__(self, name):
        print("__call__ " + name)

    def hello(self, name):
        print("hello " + name)


person = Person()
person("lzl")  # 有 __call__ 的话那么直接调用call方法
person.hello("czy")