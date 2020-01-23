
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self, dout):
        dx = dout * self.y # x와 y swap
        dy = dout * self.x

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1

#계층들
mul_appple_layer = MulLayer()
mul_tax_layer = MulLayer()

#순전파
apple_price = mul_appple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

#역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_appple_layer.backward(dapple_price)

print(price)
print(dapple, dapple_num, dtax)