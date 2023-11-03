print("***Promenljive i tipovi podataka\n")

op = "Zbir"
a = 5
b = 7
c = a + b
print("{} brojeva {} i {} je {}".format(op, a, b, c))

text = "Tekst"
integer = 10
number = 2.5 # double-precision floating point number
boolean = True # or False
complex_number = 5+3j

print("\n***Rad sa brojevima\n")

n = 5
m = 2
t = n / m # 2.5 (deljenje integera vraca float)
k = n // m # 2 (celobrojno deljenje)
r = n % m # 1 (ostatak pri deljenju)
q = n ** m # 25 (stepenovanje)
p = round(t) # 2 (zaokruzivanje)
print(n, m, t, k, r, q, p)

print("\n***Rad sa stringovima\n")

a = "Prvi red, 'abc'"
b = "Drugi red, \"abc\""
c = 'Treci red, "abc"'
d = a + '\n' + b + "\n" + c
print(d)

m = input("Unesite broj: ")
print(m, type(m))
k = int(m)
print(k, type(k))

n = 5
t = "Broj " + str(n)
print(len(t)) # 6
print(t) # Broj 5
print(t[0], type(t[0])) # B <class 'str'>
print(t[1:3]) # ro
print(t[:2]) # Br
print(t[2:]) # oj 5
print(t[-3:]) # j 5

print("\n***Liste\n")

kv = [1, 4, 9, 16, 25]
print(kv) # [1, 4, 9, 16, 25]
print(kv[0]) # 1
print(kv[-1]) # 25
print(kv[-3:]) # [9, 16, 25]
print(kv[:]) # [1, 4, 9, 16, 25]
kv = kv + [36, 49, 64, 81] # dodavanje liste
kv.append(10**2) # dodavanje elementna
print(kv) # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
kv[-3:] = [] # brise zadnja tri elementa
print(kv) # [1, 4, 9, 16, 25, 36, 49]
print(len(kv)) # 7
ml = [['a', 'b', 'c'],
 [1, 2, 3]]
print(ml[0][1]) # b
print(ml[1][:]) # [1, 2, 3]

print("\n***Višestruka dodela (fibonačijev niz)\n")

a, b = 0, 1
while a < 1000:
     print(a, end=',')
     a, b = b, a+b

print("\n***Kontrola toka (if naredba)\n")

x = int(input("Unesite celi broj: "))
if x < 0:
    x = 0
    print('Negativan broj korigovan na nulu')
elif x == 0:
    print('Nula')
else:
    print('Pozitivan broj')

if x is None:
    pass
if x is not None:
    pass
if x == 0 and (y is None or y == ''):
    pass
for i in range(10):
    if not(i >= 3 and i <= 6):
        print(i, end=' ') # 0 1 2 7 8 9


print("\n***Kontrola toka (for naredba)\n")

reci = ['knjiga', 'lopta', 'pas']
for rec in reci:
    print(rec, len(rec))
# knjiga 6
# lopta 5
# pas 3
for i in range(len(reci)):
     print(i, reci[i])
# 0 knjiga
# 1 lopta
# 2 pas
for i in range(2, 10, 3):
    print(i, end=' ')
# 2 5 8


print("\n***Funkcije\n")

def fib(n=100):
     lf = []
     a, b = 0, 1
     while a < n:
         lf.append(a)
         a, b = b, a+b
     return lf
print(fib())
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
print(fib(50))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

print("\n\n***Funkcije (argumenti)")

def ask_ok(prompt, retries=4, reminder='Please try again!'):
     while True:
         ok = input(prompt)
         if ok in ('y', 'ye', 'yes'):
            return True
         if ok in ('n', 'no', 'nop', 'nope'):
            return False
         retries = retries - 1
         if retries < 0:
            raise ValueError('invalid user response')
         print(reminder)
ask_ok('Do you really want to quit? ')
ask_ok('OK to overwrite the file? ', 2)
ask_ok('OK to overwrite the file? ',
       reminder='Come on, only yes or no!')


print("\n\n***Lamda izrazi\n")

pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[1])
print(pairs)
# [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]

def op(sign):
     if sign == '+':
        return lambda a, b: a + b
     elif sign == '-':
        return lambda a, b: a - b
     raise ValueError('Invalid operation')

plus = op('+')
minus = op('-')
print(plus(5, 3), minus(5, 3)) # 8 2


print("\n\n***Operacije nad listama\n")

ls = list() # kreira praznu listu
ls.append(-1) # dodaje element na kraj liste
print(ls) # [-1]
ls.extend(range(3)) # dodaje elemente na kraj liste na bazi iteratora
print(ls) # [-1, 0, 1, 2]
ls.insert(1, 0) # dodaje element 0 na indeksnu poziciju 1
print(ls) # [-1, 0, 0, 1, 2]
ls.remove(0) # izbacuje prvo pojavljivanje elementa
print(ls) # [-1, 0, 1, 2]
print(ls.pop(), ls) # 2 [-1, 0, 1]
print(ls.index(0)) # 1
lsr = ls.copy() # kreira i vraca kopiju liste
lsr.reverse() # invertuje redosled elemenata
print(ls, lsr) # [-1, 0, 1] [1, 0, -1]
print(['a', 'b', 'b', 'c'].count('b')) # 2 (broj pojavljivanja elementa)
kv = list(map(lambda x: x**2, range(1, 6))) # kreira listu pomocu map funkcije
print(kv) # [1, 4, 9, 16, 25]
kv = [x**2 for x in range(1, 6)] # kreira listu pomocu for naredbe
print(kv) # [1, 4, 9, 16, 25]
lt = [(x, y) for x in [1, 2, 3] for y in [3, 1, 4] if x != y]
print(lt) # [(1, 3), (1, 4), (2, 3), (2, 1), (2, 4), (3, 1), (3, 4)]


print("\n\n***N-torke (tuples)\n")

t = 12345, 54321, 'hello!' # inicijalizacija
print(t, t[0]) # (12345, 54321, 'hello!') 12345
u = t, (1, 2, 3, 4, 5) # ugnjezdavanje n-torki
print(u) # ((12345, 54321, 'hello!'), (1, 2, 3, 4, 5))
empty = ()
singleton = 'hello', # obratiti paznju na zarez na kraju
print(len(empty), empty) # 0 ()
print(len(singleton), singleton) # 1 ('hello',)
#singleton[0] = 'greska' # baca izuzetak
# TypeError: 'tuple' object does not support item assignment
t = (10, 20, 30)
x, y, z = t # raspakovanje vrednosti u promenljive
print(x, y, z) # 10 20 30

print("\n\n***Skupovi (sets)\n")

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}
print(a | b) # unija -> {1, 2, 3, 4, 5, 6, 7}
print(a & b) # presek -> {3, 4, 5}
print(a - b) # razlika -> {1, 2}
print(a ^ b) # xor -> {1, 2, 6, 7}
a = {x for x in 'abracadabra' if x not in 'abc'}
print(a) # {'r', 'd'}

print("\n\n***Rečnici (dictionaries)")

room = dict()
room['Aca'] = 331
room['Igor'] = 'M2-1'
print(room) # {'Aca': 331, 'Igor': 'M2-1’}
for key, value in room.items(): # obilazak recnika
    print(key, value)

stud = {15123: 'Петар Перић',
    15321: 'Мика Микић'}
ind = int(input('Унесите бр. индекса: '))
if ind in stud:
    print(stud[ind])
else:
    print('Студент није пронађен')

print("\n\n***Obilazak više listi (zip funkcija)\n")

indeksi = [15123, 15321]
imena = ["Pera Peric", "Mika Mikic"]
ocene = [{"OOP": 9, "SWE": 8, "CV": 10},
    {"HCI": 8, "SWE": 9}]
# Spisak studenata koji su polozici CV
for i, n, o in zip(indeksi, imena, ocene):
     if "CV" in o:
        print(i, n, o["CV"])
# 15123 Pera Peric 10

