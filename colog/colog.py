""" 
Name: Colog
Description: Making available colored and stylized text output
Author: Merock
Date: 2021.Apr.06
Version: 1.0.0
"""

class colog():
    """Making possible to print colored text or fill the background of text. Or even make your text blink!
    TextStyleSet = { 'default', 'bold', 'tiny','curvy', 'underlined', 'rareBlink', 'fastBlink', 'switchColors'}
    TextColorSet = { 'black', 'red', 'green', 'yellow', 'blue', 'purple', 'turquoise', 'white'}
    FillColorSet = { 'black', 'red', 'green', 'yellow', 'blue', 'purple', 'turquoise', 'white'}
    
    Basicly colog.log generating a  string in format like '\033[1m\033[32m\033[44m{TEXT TO PRINT}\033[0m' to print with Print Function. """

    def __init__(self, TextStyle = 'default', TextColor = 'white', FillColor = 'black', returnToDefaultMode = True):
        self.debugPrints = False

        #Codes to stylize
        self.BeginingSlashCode = '\033'
        self.AfterSlashCode = "["
        self.StartCode = self.BeginingSlashCode + self.AfterSlashCode
        self.TextStyleSet = {   'default'       :   "0",
                                'bold'          :   "1",
                                'tiny'          :   "2",
                                'curvy'         :   "3",
                                'underlined'    :   "4",
                                'rareBlink'     :   "5",
                                'fastBlink'     :   "6",
                                'switchColors'  :   "7" }
        self.TextColorSet = {   'black'         :   "30",
                                'red'           :   "31",
                                'green'         :   "32",
                                'yellow'        :   "33",
                                'blue'          :   "34",
                                'purple'        :   "35",
                                'turquoise'     :   "36",
                                'white'         :   "37"}
        self.FillColorSet = {   'black'         :   "40",
                                'red'           :   "41",
                                'green'         :   "42",
                                'yellow'        :   "43",
                                'blue'          :   "44",
                                'purple'        :   "45",
                                'turquoise'     :   "46",
                                'white'         :   "47"}
        self.EndCode = "m"

        #Generation of a string like \033[0m to be able fastly return default color\fill\style code
        self.ReturnToDefaultCode = self.StartCode + self.TextStyleSet['default'] + self.EndCode
        
        #Reading arguments or make setting by default
        if (TextColor == 'white' or TextColor not in self.TextColorSet) :
            self.ColorToSet = 'White'
        elif TextColor in self.TextColorSet:
            self.ColorToSet = TextColor

        if (FillColor == 'black' or FillColor not in self.FillColorSet):
            self.FillToSet = 'black'
        else:
            self.FillToSet = FillColor

        if (TextStyle == 'default' or TextStyle not in self.TextStyleSet):
            self.StyleToSet = 'default'
        else:
            self.StyleToSet = TextStyle

        if (TextColor != 'white' or FillColor != 'black' or TextStyle != 'default'):
            self.ReturnToDefaultMode = False
        else:
            self.ReturnToDefaultMode = returnToDefaultMode

        #Generate the code for every Color\Style\Fill variation
        for i in self.TextColorSet:
            self.TextColorSet[i] = self.StartCode + self.TextColorSet[i] + self.EndCode

        for i in self.TextStyleSet:
            self.TextStyleSet[i] = self.StartCode + self.TextStyleSet[i] + self.EndCode

        for i in self.FillColorSet:
            self.FillColorSet[i] = self.StartCode + self.FillColorSet[i] + self.EndCode
        
        if self.debugPrints:
            print("Text Color Set:", self.TextColorSet)
            print("Text Style Set:", self.TextStyleSet)
            print("Fill Color Set:", self.FillColorSet)
            print("Return To Default Mode:", self.ReturnToDefaultMode)


    def ReturnToDefaultModeSwitch(self, set, reset = False):
        self.ReturnToDefaultMode = bool(set)
        if (reset) :
           self.resetStyle()
        return True

    def resetStyle(self):
        self.StyleToSet = 'default'
        self.ColorToSet = 'white'
        self.FillToSet = 'black'
        print(self.ReturnToDefaultCode, end="")
        return True

    def setStyle(self, style = 'default', color = 'white', fill = 'black', returnToDefaultStyle = True):
        self.returnToDefaultMode = returnToDefaultStyle
        if style in self.TextStyleSet:
            self.StyleToSet = style
        if color in self.TextColorSet:
            self.ColorToSet = color
        if fill in self.FillColorSet:
            self.FillToSet = fill
        return True
    

    def variable_name(self, variable):
        name = [name for name, value in locals().items() if value is variable][0]
        return name

    def green_red_print(self, value, text, style = 'untouch', color = 'untouch', fill = 'untouch'):
        if value:
            self.print(f'{text} is True', color = 'green')
        else:
            self.print(f'{text} is False', color = 'red')

    def print(self, text, style = 'untouch', color = 'untouch', fill = 'untouch'):
        
        if style != 'untouch' and style in self.TextStyleSet:
            self.StyleToSet = style
        if fill != 'untouch' and fill in self.FillColorSet:
            self.FillToSet = fill
        if color != 'untouch' and color in self.TextColorSet:
            self.ColorToSet = color
        if self.debugPrints:
            print("Style:", self.StyleToSet)
            print("Color:", self.ColorToSet)
            print("Fill:", self.FillToSet)
            print("RTDM:", self.ReturnToDefaultMode)

        styleString = ""
        
        styleString = styleString + self.TextStyleSet[self.StyleToSet.lower()]
        styleString = styleString + self.TextColorSet[self.ColorToSet.lower()]
        styleString = styleString + self.FillColorSet[self.FillToSet.lower()]

        styleString = styleString + "{}"
        if self.ReturnToDefaultMode == True:
            styleString = styleString + self.ReturnToDefaultCode

        if self.debugPrints:
            print(styleString + text)

        #Expected sting in format "\033[0m\033[32mTextToPrint"
        #Or expected "\033[0m\033[32mTextToPrint\033[0m" if ReturnToDefaultMode is True
        print(styleString.format(text))
        print(self.ReturnToDefaultCode, end="")
        if (self.ReturnToDefaultMode) :
            self.resetStyle()
        
        
if __name__ == '__main__':
    c = colog()
    c.print("Curvy green text on red bg", style = "curvy", color = "green", fill = "red")
    c.print("Tiny red text on default BG", style = "tiny", color = "red")
    c.print("Bold default colored text on default BG", style = "bold")
    c.ReturnToDefaultModeSwitch(False)
    c.print("Tiny red text on default BG, its switching return to default mode OFF", style = "tiny", color = "red")
    c.print("SEE? SEE? SEE?")
    c.ReturnToDefaultModeSwitch(set = True, reset = True)
    c.print("Ok, I turned the switch back, for you.")
    c.print("Vivod")
