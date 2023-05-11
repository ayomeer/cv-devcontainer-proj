import lib.cppmodule as cpp 

obj = cpp.CppHomography()
obj.setMember(2)

print("memberVar: ", obj.getMember())

obj.publicVar = 3

print("publicVar: ", obj.publicVar)