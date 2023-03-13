# Zadanie 1
# Zdefiniuj funkcję zwracającą wartość logiczną prawda jeżeli hasło przekazane do niej jako parametr tekstowy jest silne, a wartość logiczną fałsz - w przeciwnym przypadku.
# Hasło jest silne jeżeli jednocześnie spełnione są wszystkie poniższe warunki:
# - ma długość co najmniej dziesięciu znaków
# - zawiera co najmniej jedną dużą literę
# - zawiera co najmniej jedną małą literę
# - zawiera co najmniej jedną cyfrę
# - zawiera co najmniej jeden znak specjalny z grupy: @ $ _
#
# Podpowiedź: wykorzystaj metody islower(), isupper() i isdigit() obiektów typu tekstowego
#
# Tekst odpowiedzi
# #
# def StrongPassowrd(hasło):
#     text =len(hasło)
#     if text >= 10:
#         for i in hasło:
#             if i in i.islower():


#     else:
#         return False
#
#
# print(StrongPassowrd('Chujkkjfjefer'))

#
# # co najmniej jedna duża litera
# def Hasło(hasło):
#     for i in hasło:
#         if i.isupper():
#             return True
#     return False
#
#
# print(Hasło('huj'))
#
#
# # co najmniej jedna mała litera
# def Hasło(hasło):
#     for i in hasło:
#         if i.lower():
#             return True
#     return False
#
#
# # co najmnien 1 liczba
# def Hasło(hasło):
#     for i in hasło:
#         if i.digit():
#             return True
#     return False


# def StrongPassowrd(hasło):
#     text = len(hasło)
#     if text >= 10:
#         for i in hasło:
#             if i.lower():
#                 return True
#             for i in hasło:
#                 if i.digit():
#                     return True
# return 'Silne hasło'
#
# # print(StrongPassowrd('hdbKKKKdab12'))
#
#
# def Parametry(x, y, z):
#     nowy_zbiór = []
#     for i in x:
#         if i in x and i in y and i in z:
#             nowy_zbiór.append(i)
#     return nowy_zbiór
#
#
# print(Parametry([5, 6, 7, 8], [2, 3, 5, 8], [3, 4, 5, 8]))
#
#
# def Parametry(x, y, z):
#     if not x or not y or not z:
#         zbior = "x" if not x else "y" if not y else "z"
#         raise ValueError(f"Błąd: zbiór {zbior} jest pusty!")
#
#     nowy_zbiór = []
#     for i in x:
#         if i in x and i in y and i in z:
#             nowy_zbiór.append(i)
#     return nowy_zbiór
#
#
# print(Parametry([], [2, 3, 5, 8], [3, 4, 5, 8]))
#
# # A jakbym chciała wszystko


def Wspolne(x, y, z):
    wspólne_dla_wszystkich = []
    wspólne_x_y = []
    wspólne_y_z = []
    wspólne_x_z = []

    for i in x:
        if i in y and i in z:
            wspólne_dla_wszystkich.append(i)
        if i in y:
            wspólne_x_y.append(i)
        if i in z:
            wspólne_x_z.append(i)

    for i in y:
        if i in x and i in z and i not in wspólne_dla_wszystkich:
            wspólne_dla_wszystkich.append(i)
        if i in z and i not in wspólne_x_z:
            wspólne_y_z.append(i)

    return (wspólne_dla_wszystkich, wspólne_x_y, wspólne_y_z, wspólne_x_z)

print(Wspolne([2,3,4,5],[2,4,7,8],[2,5,8,0]))
print("Hello")