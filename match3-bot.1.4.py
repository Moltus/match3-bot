import numpy as np
from PIL import ImageGrab
import cv2 as cv
import pyautogui
import time
import sys
import json
from random import choice

screen_resolution = (1920,1080)
game_roi = None


class PreCapture:
    def __init__(self):

        self.halfscrn_cap = None
        self.region_of_interest = None
        self.autodetect = False
        self.roi_preview = None
        self.saved_roi = self.get_saved_roi()

        if self.saved_roi:
            print("Press -enter- to validate previously saved region\n",
                "\bPress -space- to get a new region with auto-detection\n",
                "\bPress -escape- to quit")

        else:
            print("Press -enter- to validate region\n",
                  "\bPress -escape- to quit")
        

    def main_loop(self):
        
        while True:
            # capture current overall right-half screen
            self.halfscrn_cap = np.array(ImageGrab.grab(bbox=(
                    screen_resolution[0]/2+50, 0,
                    screen_resolution[0], screen_resolution[1]-50)))
            if self.autodetect:
                self.roi_preview = self.get_roi_preview()
            else:
                self.roi_preview = self.saved_roi

            if self.roi_preview:
                cv.rectangle(self.halfscrn_cap, tuple(self.roi_preview[:2]),
                             tuple(self.roi_preview[2:]), (0,255,0), 2)
            else:
                pass

            cv.imshow('screen catpure', self.halfscrn_cap)

            k = cv.waitKey(25) & 0xFF
            if k == 27:
                print("Pressed -escape- > Bye bye.")
                cv.destroyAllWindows()
                sys.exit(0)
            elif k == 13:
                if self.roi_preview:
                    self.save_roi()
                    print("Pressed -Enter- > region validated")
                    break
                else:
                    print("Pressed -Enter- > Couldn't find any region to capture")
            elif k == 32 and self.autodetect == False:
                self.autodetect = True
                print("Pressed -space- > switching to autodetect mode")   
                

        cv.destroyAllWindows()
        return self.region_of_interest

    def get_saved_roi(self):
        try:
            with open("saved_roi.json", 'r') as read_file:
                return json.load(read_file)
        except FileNotFoundError:
            print("Couldn't find an existing saved region of interest")
            print("Starting auto-detection")
            self.autodetect = True


    def get_roi_preview(self):

        # create 2 kernels for topleft and botright corners
        # of selection rectangle
        kernel_TLcorner = np.zeros((51,51), dtype='int')
        kernel_TLcorner[23:25, 23:51] = -1
        kernel_TLcorner[23:51, 23:25] = -1
        kernel_TLcorner[25:51, 25:51] = 1
        # inverted topleft gets us botright
        kernel_BRcorner = kernel_TLcorner[::-1,::-1]

        # convert catpured screen to gray to get b&w threshold
        gray = cv.cvtColor(self.halfscrn_cap, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray,50,255,cv.THRESH_BINARY)

        # filter the screen with kernels to get the corners
        hitmiss = cv.morphologyEx(thresh, cv.MORPH_HITMISS, kernel_TLcorner) \
                  + cv.morphologyEx(thresh, cv.MORPH_HITMISS, kernel_BRcorner)

        # exclude top left and bottom right corners of the filtered screen
        hitmiss[0:2,0:2] = 0
        hitmiss[self.halfscrn_cap.shape[0]-2:self.halfscrn_cap.shape[0],
                   self.halfscrn_cap.shape[1]-2:self.halfscrn_cap.shape[0]] = 0

        # get coordinates and draw rectangle of interest over screen
        # tuple(zip) method to extract only coords
        rect_indices = np.where(hitmiss == 255)
        rect_coords = tuple(zip(rect_indices[0], rect_indices[1]))

        if len(rect_coords) == 2:
            # extract from tuple coordinates for visual cue rectangle
            # invert x and y with negative increment
            rect_TL = rect_coords[0][::-1]
            rect_BR = rect_coords[1][::-1]
            roi_preview = [int(i) for i in rect_TL] + [int(j) for j in rect_BR]
            print(roi_preview)
            return roi_preview
        else:
            pass

    def save_roi(self):
        # get region of interest real screen coordinates
        # by adding half screen horizontally and +50/50 pix safe zone
        print("roi_preview :", self.roi_preview)
        with open("saved_roi.json", 'w') as write_file:
            json.dump(self.roi_preview, write_file)

        self.region_of_interest = [
            self.roi_preview[0] + int(screen_resolution[0]/2+50),
            self.roi_preview[1],
            self.roi_preview[2] + int(screen_resolution[0]/2+50),
            self.roi_preview[3]
            ]

        

class RoICapture:
    def __init__(self):
        self.roi_cap = None
        self.roi_toggle = True
        self.bot_toggle = False
        self.tiles_roi, self.skills_roi, self.lifebars_roi, self.plcmt_roi = (
            self.get_regions())
        self.tiles_val = np.zeros((8,8), dtype=np.uint8)
        self.skills_val = np.zeros((3,4), dtype=bool)
        self.teamup_toggle = False
        self.lifebars = [True, True, True, True, True, True]
        self.player_turn = False
        self.skill_plcmt = False


        print("Captured region coordinates are :", game_roi)

        print("Press -r- to show/hide captured regions\n",
              "Press -t- to activate/deactivate teamup's skills\n",
              "Press -enter- to start/stop bot\n",
              "Press -d- for debugging info\n",
              "Press -s- to take a screenshot\n",
              "Press -escape- to quit")


    def main_loop(self):

        while True:
            self.roi_cap = np.array(ImageGrab.grab(bbox=(game_roi)))
            # cv.imwrite('RoI_capture.png', self.roi_cap)

            self.get_regions_info()

            if self.bot_toggle:
                game_bot.input(self.lifebars, self.skills_val, self.tiles_val,
                               self.player_turn, self.skill_plcmt)

            else:
                pass

            if self.roi_toggle:
                self.show_regions(self.tiles_roi, self.skills_roi,
                                  self.lifebars_roi, self.plcmt_roi)
            else:
                cv.destroyWindow('Regions of interest')

            self.show_color_scrn()

            k = cv.waitKey(25) & 0xFF
            if k == 27: # key = escape
                print("Pressed -escape- > Bye bye.")
                break
            elif k == ord('r'): # key = r
                self.roi_toggle = not self.roi_toggle
                print(
                "Pressed -r- :", "Showing" if self.roi_toggle else "Hiding",
                "regions of interest"
                )

            elif k == 13: # key = enter

                if np.any(self.tiles_val == 0) and not self.bot_toggle:
                    print(
                    "Pressed -enter- > Can't start bot without a complete board"
                    )
                else:
                    self.bot_toggle = not self.bot_toggle
                    print(
                    "Pressed -enter- > Bot is", "ON" if self.bot_toggle else "OFF"
                    )
            elif k == ord('d'): # debug
                print("Pressed -d- > Debug info:")
                print("grid colors :\n", self.tiles_val)
                print("skills :\n", self.skills_val)
                print("player turn :", self.player_turn)
                print("player's life :", self.lifebars[:3])
                print("enemy's life :", self.lifebars[3:])
                print("skill placement :", self.skill_plcmt)

            elif k == ord('t'): # key = t
                self.teamup_toggle = not self.teamup_toggle
                print(
                "Pressed -t- > Teamups are", "ON" if self.teamup_toggle else "OFF"
                )

            elif k == ord('s'): # key = s
                cv.imwrite('capture.png', self.roi_cap)
                print("Pressed -s- > Screen captured as 'capture.png'")


        cv.destroyAllWindows()


    def get_regions(self):
        ht = game_roi[3] - game_roi[1]
        wd = game_roi[2] - game_roi[0]


        def get_lifebars_regions():
            lifebars_roi = []
            # get line dimensions
            y = int(ht*0.042)
            a = int(wd*0.007), int(wd*0.091)
            b = int(wd*0.139), int(wd*0.240)
            c = int(wd*0.289), int(wd*0.403)
            d = int(wd*0.594), int(wd*0.713)
            e = int(wd*0.767), int(wd*0.867)
            f = int(wd*0.917), int(wd*0.998)

            for i in a,b,c,d,e,f:
                lifebars_roi.append((i[0], y, i[1], y))

            return lifebars_roi

        def get_plcmt_region():
            y = int(ht*0.14)
            x1, x2 = int(wd*0.2), int(wd*0.8)
            plcmt_roi = (x1, y, x2, y)
            return plcmt_roi


        def get_skills_regions():
            skills_roi = np.zeros((3,4,4), int)
            # get line dimensions
            line_width = wd*0.075
            left_shift = wd*0.01
            line_v_spacing = ht*0.018
            line_h_spacing = wd*0.2475
            x1 = int(wd*0.178)
            y = int(ht*0.313)
            x2 = int(x1 + line_width)

            for i in range(4):
                for j in range(3):
                    skills_roi[j,i] = (
                        (x1 + int(line_h_spacing*i) - int(left_shift*j)),
                        (y + int(line_v_spacing*j)),
                        (x2 + int(line_h_spacing*i) - int(left_shift*j)),
                        (y + int(line_v_spacing*j))
                    )
            return skills_roi


        def get_tiles_regions():
            tiles_roi = np.zeros((8,8,4), int)
            # get squares dimensions
            box_width = ht*0.029
            box_spacing = wd*0.1205
            TLx = int(wd*0.054)
            TLy = int(ht*0.422)
            BRx = int(TLx + box_width)
            BRy = int(TLy + box_width)


            for i in range(8):
                for j in range(8):
                    tiles_roi[j,i] = (
                        (TLx + int(box_spacing*i)),
                        (TLy + int(box_spacing*j)),
                        (BRx + int(box_spacing*i)),
                        (BRy + int(box_spacing*j))
                        )
            return tiles_roi


        return (
            get_tiles_regions(), get_skills_regions(), get_lifebars_regions(),
            get_plcmt_region()
        )

    def show_regions(self, tiles_roi, skills_roi, lifebars_roi, plcmt_roi):
        for i in range(8):
            for j in range(8):
                [cv.rectangle(self.roi_cap, (tiles_roi[i,j,0], tiles_roi[i,j,1]),
                              (tiles_roi[i,j,2], tiles_roi[i,j,3]), (0,255,0), 2)]

        for i in range(4):
            for j in range(3):
                [cv.line(self.roi_cap, (skills_roi[j,i,0], skills_roi[j,i,1]),
                              (skills_roi[j,i,2], skills_roi[j,i,3]), (0,255,0), 2)]

        for i in lifebars_roi:
            cv.line(self.roi_cap, (i[0], i[1]), (i[2], i[3]), (0,255,0), 2)

        cv.line(self.roi_cap, plcmt_roi[:2], plcmt_roi[2:], (0,255,0), 2)

        cv.imshow('Regions of interest', self.roi_cap)

    def get_regions_info(self):

        def get_lifebars_status():
            for x in range(6):
                x1, y1, x2, _ = self.lifebars_roi[x]
                roi = self.roi_cap[y1, x1:x2]
                # print(roi) # debug
                self.lifebars[x] = (any((n[0]>=205 for n in roi)))

        def get_plcmt_status():
            x1, y1, x2, _ = self.plcmt_roi
            self.skill_plcmt = np.all(self.roi_cap[y1, x1:x2] == 236)   


        def get_skills_status():
   
            for x in range(4):
                for y in range(3):
                    x1, y1, x2, _ = self.skills_roi[y, x]
                    # print(x1, y1 , x2)
                    roi = self.roi_cap[y1, x1:x2]
                    # print("skill roi:", x, y, roi)
                    self.skills_val[y, x] = (
                        any((all(j>200 for j in roi[:,i]) for i in range(3)))
                    ) or (np.all(roi == 75))

            if not self.teamup_toggle:
                self.skills_val[:,3] = False
            else:
                pass


        def get_tiles_color():
            # get board colors by the mean of their region hsv values
            for x in range(8):
                for y in range(8):
                    TLx, TLy, BRx, BRy = self.tiles_roi[x, y]
                    roi = cv.cvtColor(self.roi_cap[TLy:BRy, TLx:BRx], cv.COLOR_RGB2HSV)
                    h, s, v, _ = (int(i) for i in cv.mean(roi))
                    # print(x, y, "h, s, v :", h, s, v) #debug

                    if s > 60 and 0 <= h < 15:
                        self.tiles_val[x,y] = 1
                    elif s > 60 and 20 < h < 40:
                        self.tiles_val[x,y] = 2
                    elif s > 60 and 45 < h < 65:
                        self.tiles_val[x,y] = 3
                    elif s > 60 and 105 < h < 115:
                        self.tiles_val[x,y] = 4
                    elif s > 60 and 140 < h < 155:
                        self.tiles_val[x,y] = 5
                    elif s < 32 and h < 30:
                        self.tiles_val[x,y] = 6
                    elif 60 < s < 70 and 165 < v < 180:
                        self.tiles_val[x,y] = 7
                    else:
                        self.tiles_val[x,y] = 0
                        # print(f"{x, y}, h : {h}, s : {s}, v : {v}") # debug

        def get_bottom_pix():
            # get rgb value for a pixel right under board
            # mainly to check if it's player's turn
            wd = game_roi[2] - game_roi[0]
            ht = game_roi[3] - game_roi[1]
            x = int(wd / 2)
            y = int(ht * 0.96)
            pix = self.roi_cap[y, x]
            if  pix[0] < 5 and 35 < pix[1] < 45 and 105 < pix[2] < 115:
                self.player_turn = True
            else:
                self.player_turn = False

        get_lifebars_status()
        get_skills_status()
        get_tiles_color()
        get_bottom_pix()
        get_plcmt_status()


    def show_color_scrn(self):
        color_preview = np.zeros((8,8,3), np.uint8)
        int_to_color = {
        0:(125,125,125), 1:(255,0,0), 2:(255,255,0), 3:(0,255,0),
        4:(0,0,255), 5:(255,0,255), 6:(0,0,0), 7:(245, 245, 245)
                       }
        for x in range(8):
            for y in range(8):
                color_preview[x,y] = int_to_color[self.tiles_val[x,y]]

        color_preview = cv.cvtColor(color_preview, cv.COLOR_BGR2RGB) #debug
        # cv.imwrite('preview_colors.png',color_preview)

        resized_preview = cv.resize(color_preview, (300, 300),
                                    interpolation=cv.INTER_NEAREST)
        cv.imshow('colors captured', resized_preview)

class Bot:
    def __init__(self):
        # self.player_life = lifebars[:3]
        # self.enemy_life = lifebars[3:]

        # self.lifebars_delay = 4
        self.lifebars_check_time = None
        
        # self.action_timer = 1.5
        self.last_action_time = time.time()

        # self.player_action = True

        # skill_check is here to ensure once detected a readied skill is exec
        # self.skill_check = False


    def input(self, lifebars, skills_val, tiles_val, player_turn, plcmt):
        player_life = lifebars[:3]
        enemy_life = lifebars[3:]
        self.skills = skills_val
        
        if plcmt == False and ((not any(enemy_life)) or (not any(player_life))):
            if self.get_lifebars_timer(4):
                self.game_over('win' if player_life else 'lost')
            else:
                pass

        
        if player_turn:
            if plcmt:
                    print("skill tile placement :", plcmt)
                    self.random_tile(tiles_val)
            elif self.get_action_timer(1):               
                if np.any(skills_val):
                    self.skill_use(skills_val)
                    # self.last_action_time = time.time()
                    
                elif np.any(tiles_val):
                    self.tile_flip(tiles_val)  
                    # self.last_action_time = time.time()
                else:
                    game_capture.bot_toggle = False
                    print("\aBot is OFF : couldn't find any move to do!")
            else:
                pass
            
        else:
            pass
        

    def get_lifebars_timer(self, delay):
        if self.lifebars_check_time:
            delay = time.time() - self.lifebars_check_time
            if 6 > delay > delay:
                return True
            elif delay >= 6:
                self.lifebars_check_time = time.time()
                return False
            else:
                return False
        else:
            self.lifebars_check_time = time.time()
            return False

    def random_tile(self, tiles_val):

        possible_tiles = tuple(zip(*np.where(tiles_val!=0)))
        tile = choice(possible_tiles)
        mouse.click_tile(tile)
        

    def get_action_timer(self, delay):
        if self.last_action_time:
            if (time.time() - self.last_action_time) > delay:
                self.last_action_time = None
                return True
            else:
                return False
        else:
            self.last_action_time = time.time()
            return False

    def skill_use(self, skills_val):      
            for i in range(4):
                for j in range(3):
                    if skills_val[j,i]:
                        mouse.click_skill((j,i))
                        return

    def tile_flip(self, tiles_val):
        current_board = Board(tiles_val)
        coords = current_board.get_best_move()
        mouse.drag_tile(coords)


    def game_over(self, outcome):
        # time.sleep(1)
        
        if outcome == 'win':
            game_capture.bot_toggle = False
            print("\aBot is OFF : enemy's life bars are empty!")
            print("You win!")
        elif outcome == 'lost':
            game_capture.bot_toggle = False
            print("\aBot is OFF : player's life bars are empty!")
            print("\aYou lost!")
        else:
            pass





class Board:
    def __init__(self, board):
        self.current_board = board

    def h_flip(self, y, x):
        tmp = self.current_board.copy()

        if (0 <= x < 7) and (tmp[y,x] != tmp[y,x+1]):
            tmp[y, x:x+2] = tmp[y, x:x+2][::-1]
        else:
            pass

        left_tile_chain = self.get_neighbors(y, x, tmp)
        right_tile_chain = self.get_neighbors(y, x+1, tmp)

        nb_chains = (left_tile_chain > 0) + (right_tile_chain > 0)
        max_chain = max(left_tile_chain, right_tile_chain)

        if nb_chains != 0:
            return max_chain, nb_chains


    def v_flip(self, y, x):
        tmp = self.current_board.copy()

        if (0 <= y < 7) and (tmp[y,x] != tmp[y+1,x]):
            tmp[y:y+2, x] = tmp[y:y+2, x][::-1]
        else:
            pass

        up_tile_chain = self.get_neighbors(y, x, tmp)
        down_tile_chain = self.get_neighbors(y+1, x, tmp)

        nb_chains = (up_tile_chain > 0) + (down_tile_chain > 0)
        max_chain = max(up_tile_chain, down_tile_chain)

        if nb_chains != 0:
            return max_chain, nb_chains


    def get_neighbors(self, y, x, temp_board):
        tmp = temp_board
        chain = {(y,x)}
        color = tmp[y,x]
        # find max connected tiles of same color on horizontal
        def get_h_nbr():
            if len(tmp[y, x-1:x+2]) == 3 and np.all(tmp[y, x-1:x+2] == color):
                chain.add((y, x-1))
                chain.add((y, x+1))
            if len(tmp[y, x:x+3]) == 3 and np.all(tmp[y, x:x+3] == color):
                chain.add((y, x+1))
                chain.add((y, x+2))
            if len(tmp[y, x-2:x+1]) == 3 and np.all(tmp[y, x-2:x+1] == color):
                chain.add((y, x-2))
                chain.add((y, x-1))

        # find max connected tiles of same color on vertical
        def get_v_nbr():
            if len(tmp[y-1:y+2, x]) == 3 and np.all(tmp[y-1:y+2, x] == color):
                chain.add((y-1, x))
                chain.add((y+1, x))
            if len(tmp[y:y+3, x]) == 3 and np.all(tmp[y:y+3, x] == color):
                chain.add((y+1, x))
                chain.add((y+2, x))
            if len(tmp[y-2:y+1, x]) == 3 and np.all(tmp[y-2:y+1, x] == color):
                chain.add((y-2, x))
                chain.add((y-1, x))

        get_h_nbr()
        get_v_nbr()

        return len(chain) if len(chain) >= 3 else 0


    def find_moves(self):
        moves = {}
        for y in range(8):
            for x in range(8):
                h_res = self.h_flip(y,x) if x!=7 else None
                if h_res:
                    moves[((y,x),(y,x+1))] = h_res

                v_res = self.v_flip(y,x) if y!=7 else None
                if v_res:
                    moves[((y,x),(y+1,x))] = v_res

        return moves

    def get_best_move(self):
        moves = self.find_moves()
        max = (0,0)
        best = None
        for k, v in moves.items():
            if v >= max:
                max = v
                best = k
            else:
                pass
        # print(f"best move is {best}") # debug
        return best


class MouseControl:

    def __init__(self):
        ht = game_roi[3] - game_roi[1]
        wd = game_roi[2] - game_roi[0]
        # top left tile position and tile spacing
        self.t0 = game_roi[0] + int(wd*0.082), game_roi[1] + int(ht*0.436)
        self.tile_spacing = int(wd*0.12)
        # left skill big button position and skill spacing
        self.s0 = game_roi[0] + int(wd*0.089), game_roi[1] + int(ht*0.332)
        self.skill_spacing = int(wd*0.25)
        # get a grid with all click politions for color skill buttons
        self.radialskill = np.zeros((3,4,2))
        # fill grid with tuples of x,y for every skill
        self.radialskill[0,0] = game_roi[0] + int(wd*0.183), game_roi[1] + int(ht*0.555)
        self.radialskill[1,0] = game_roi[0] + int(wd*0.39), game_roi[1] + int(ht*0.51)
        self.radialskill[2,0] = game_roi[0] + int(wd*0.5), game_roi[1] + int(ht*0.415)
        self.radialskill[0,1] = game_roi[0] + int(wd*0.358), game_roi[1] + int(ht*0.555)
        self.radialskill[1,1] = game_roi[0] + int(wd*0.565), game_roi[1] + int(ht*0.51)
        self.radialskill[2,1] = game_roi[0] + int(wd*0.675), game_roi[1] + int(ht*0.415)
        self.radialskill[0,2] = game_roi[0] + int(wd*0.59), game_roi[1] + int(ht*0.555)
        self.radialskill[1,2] = game_roi[0] + int(wd*0.797), game_roi[1] + int(ht*0.51)
        self.radialskill[2,2] = game_roi[0] + int(wd*0.907), game_roi[1] + int(ht*0.415)
        # TODO : 4th column (teamups)

        # validate tile choice button coordinates
        self.button = game_roi[0]+int(wd*0.5), game_roi[1]+int(ht*0.2)


    def click_skill(self, coords):
        print("click skill at x, y :", coords)
        ax = self.s0[0] + coords[1] * self.skill_spacing
        ay = self.s0[1]
        # print("ax, ay :", ax, ay)
        pyautogui.moveTo(ax, ay, 0.25)
        pyautogui.click(duration=0.25)
        bx, by = self.radialskill[coords[0], coords[1]]
        # print("bx, by :", bx, by)
        pyautogui.moveTo(bx, by, 0.25)
        # pyautogui.click(clicks=2, interval=0.25)
        pyautogui.click(clicks=2, interval=0.1)

    def click_tile(self, coords):
        # click a random tile then click the validate button
        # tile_coordinates:
        tx = self.t0[0] + int(coords[1] * self.tile_spacing)
        ty = self.t0[1] + int(coords[0] * self.tile_spacing)
        
        pyautogui.moveTo(tx, ty, 0.5)
        pyautogui.click()
        pyautogui.moveTo(self.button[0], self.button[1])
        pyautogui.click()


    def drag_tile(self, tiles_coords):
        
        ax = self.t0[0] + tiles_coords[0][1] * self.tile_spacing
        ay = self.t0[1] + tiles_coords[0][0] * self.tile_spacing
        bx = self.t0[0] + tiles_coords[1][1] * self.tile_spacing
        by = self.t0[1] + tiles_coords[1][0] * self.tile_spacing
        # print("mouse movement", ax,ay,bx,by) # debug
        pyautogui.moveTo(ax, ay, 0.25)
        pyautogui.dragTo(bx, by, 0.25, button='left')

        # pyautogui.moveTo(self.tx0 + self.tile_spacing, self.ty0 + self.tile_spacing, 1)



capture = PreCapture()
game_roi = capture.main_loop()
mouse = MouseControl()
game_capture = RoICapture()
game_bot = Bot()
game_capture.main_loop()
