import numpy as np
import cv2
from scipy.ndimage import interpolation as inter
from PIL import Image as im
from skimage.morphology import skeletonize

###############################################################################

def deskew(image,filter:int=1):


    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsus Binarization
    if filter != 0:
        blur = cv2.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # deskew

    bin_img = (binary_img // 255.0)

    delta = 0.1
    limit = 3
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        data = inter.rotate(bin_img, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = im.fromarray((255 * data).astype("uint8"))

    pix = np.array(img)

    return pix


###############################################################################

def get_lines(image, cut=3):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bitwise_not(gray_img)

    clean_img = deskew(gray_img, 0)

    lines = []
    start = -1
    cnt = 0

    projection_bins = np.sum(clean_img, 1).astype('int32')
    
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                lines.append(clean_img[max(start-1, 0):idx, :])
                cnt = 0
                start = -1

    return lines

###############################################################################

def get_words(line_image, cut=3):
    
    line_words = []
    start = -1
    cnt = 0
   
    projection_bins = np.sum(line_image, 0).astype('int32')
        
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                line_words.append(line_image[:, max(start-1, 0):idx])
                cnt = 0
                start = -1
   
    line_words.reverse()
    
    return line_words


###############################################################################
###############################################################################
###############################################################################

def horizontal_transitions(word_img, baseline_idx):
    
    max_transitions = 0
    line_idx = baseline_idx-1
    lines = []
    # new temp image with no dots above baseline
    
    while line_idx >= 0:
        current_transitions = 0
        flag = 0

        horizontal_line = word_img[line_idx, :]
        for pixel in reversed(horizontal_line):

            if pixel == 1 and flag == 0:
                current_transitions += 1
                flag = 1
            elif pixel == 0 and flag == 1:
                current_transitions += 1
                flag = 0
                
        if current_transitions >= max_transitions:
            max_transitions = current_transitions
            lines.append(line_idx)

        line_idx -= 1
    
    return lines[len(lines)//2]

###############################################################################

def vertical_transitions(word_img, cut):
    
    transitions = 0

    vertical_line = word_img[:, cut]

    flag = 0
    for pixel in vertical_line:

        if pixel == 1 and flag == 0:
            transitions += 1
            flag = 1
        elif pixel == 0 and flag == 1:
            transitions += 1
            flag = 0

    return transitions

###############################################################################

def cut_points(word_img, VP, MFV, MTI, baseline_idx):
      
    # flag to know the start of the word
    f = 0

    flag = 0
    (h, w) = word_img.shape
    i = w-1
    separation_regions = []

    wrong = 0
    # loop over the width of the image from right to left
    while i >= 0:

        pixel = word_img[MTI, i]
        
        if pixel == 1 and f == 0:
            f = 1
            flag = 1

        if f == 1:

            # Get start and end of separation region (both are black pixels <----)
            if pixel == 0 and flag == 1:
                start = i+1
                flag = 0
            elif pixel == 1 and flag == 0:
                end = i         # end maybe = i not i+1
                flag = 1

                mid = (start + end) // 2

                left_zero = -1
                left_MFV = -1
                right_zero = -1
                right_MFV = -1
                # threshold for MFV
                T = 1

                j = mid - 1
                # loop from mid to end to get nearest VP = 0 and VP = MFV
                while j >= end:
                    
                    if VP[j] == 0 and left_zero == -1:
                        left_zero = j
                    if VP[j] <= MFV + T and left_MFV == -1:
                        left_MFV = j

                    # if left_zero != -1 and left_MFV != -1:
                    #     break

                    j -= 1

                j = mid
                # loop from mid to start to get nearest VP = 0 and VP = MFV
                while j <= start:

                    if VP[j] == 0 and right_zero == -1:
                        right_zero = j
                    if VP[j] <= MFV + T and right_MFV == -1:
                        right_MFV = j

                    if right_zero != -1 and right_MFV != -1:
                        break

                    j += 1

                # Check for VP = 0 first
                if VP[mid] == 0:
                    cut_index = mid
                elif left_zero != -1 and right_zero != -1:
                    
                    if abs(left_zero-mid) <= abs(right_zero-mid):
                        cut_index = left_zero
                    else:
                        cut_index = right_zero
                elif left_zero != -1:
                    cut_index = left_zero
                elif right_zero != -1:
                    cut_index = right_zero

                # Check for VP = MFV second
                # elif VP[mid] <= MFV+T:
                #     cut_index = mid
                elif left_MFV != -1:
                    cut_index = left_MFV
                elif right_MFV != -1:
                    cut_index = right_MFV
                else:
                    cut_index = mid


                seg = word_img[:, end:start]
                HP = np.sum(seg, 1).astype('int32')
                SHPA = np.sum(HP[:MTI])
                SHPB = np.sum(HP[MTI+1:])
                
                top = 0
                for idx, proj in enumerate(HP):
                    if proj != 0:
                        top = idx
                        break

                cnt = 0
                for k in range(end, cut_index+1):
                    if vertical_transitions(word_img, k) > 2:
                        cnt = 1
                if SHPB == 0 and (baseline_idx - top) <= 5 and cnt == 1:
                    # breakpoint()
                    wrong = 1
                else:
                    separation_regions.append((end, cut_index, start))

        i -= 1

    return separation_regions, wrong

###############################################################################

def check_baseline(word_img, start, end, upper_base, lower_base):
    
    j = end+1

    cnt = 0
    while j < start:
    
        # Black pixel (Discontinuity)
        base = upper_base
        while base <= lower_base:
            
            pixel = word_img[base][j]
            cnt += pixel

            base += 1
        
        j += 1

    if cnt == 0:
        return False

    return True

###############################################################################

def inside_hole(word_img, end_idx, start_idx):
    '''Check if a segment has a hole or not'''

    if end_idx == 0 and start_idx == 0:
        return 0

    sk = skeletonize(word_img)
    j = end_idx + 1
    flag = 1
    while j < start_idx:
        VT = vertical_transitions(sk, j)
        if VT <= 2:
            flag = 0
            break
        j += 1
    
    return flag

###############################################################################

def check_hole(segment):
    '''Check if a segment has a hole or not'''

    # no_dots = segment.copy()

    contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = 0
    for hier in hierarchy[0]:
        if hier[3] >= 0:
            cnt += 1

    return cnt != 0

###############################################################################

def remove_dots(word_img, threshold=11):

    no_dots = word_img.copy()

    components, labels, stats, GoCs = cv2.connectedComponentsWithStats(no_dots, connectivity=8)
    char = []
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if size > threshold:
            char.append(label)
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if label not in  char:
            no_dots[labels == label] = 0

    return no_dots

###############################################################################

def check_dots(segment):

    contours, hierarchy = cv2.findContours(segment[:, 1:segment.shape[1]-1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = 0
    for c in contours:
        if len(c) >= 1:
            cnt +=1 
    return cnt > 1

###############################################################################

def check_stroke(no_dots_copy, segment, upper_base, lower_base, SR1, SR2):

    T = 1
    components, labels, stats, cen= cv2.connectedComponentsWithStats(segment, connectivity=8)
    skeleton = skeletonize(segment.copy()).astype(np.uint8)
    (h, w) = segment.shape

    cnt = 0
    for label in range(1, components):
        if stats[label][4] > 3:
            cnt += 1
        else:
            segment[labels==label] = 0

    if cnt > 2 or cnt == 0:
        return False

    if check_hole(segment) or inside_hole(no_dots_copy, SR1[0], SR1[1]) or inside_hole(no_dots_copy, SR2[0], SR2[1]):
        return False

    HP = np.sum(skeleton, 1).astype('int32')
    VP = np.sum(segment, 0).astype('int32')

    seg_l = -1
    seg_r = -1
    for i in range(0, len(VP)):
        if VP[i] != 0:
            seg_l = i
            break
    for i in range(len(VP)-1, -1, -1):
        if VP[i] != 0:
            seg_r = i
            break

    seg_width = seg_r - seg_l + 1
    SHPA = np.sum(HP[:upper_base])
    SHPB = np.sum(HP[lower_base+T+1:])
    MFV_HP = np.argmax(np.bincount(HP)[1:])+1
    MFV = lower_base - upper_base + 1 + T

    top_pixel = -1
    for i, proj in enumerate(HP):
        if proj != 0:
            top_pixel = i
            break
    height = upper_base-top_pixel
    
    VT = 0
    for i in range(w):
        if vertical_transitions(skeleton, i) > 2:
            VT += 1
    cnt = 0
    for proj in VP:
        if proj >= height:
            cnt += 2
        elif proj == height-1:
            cnt += 1
    # abs(MFV - MFV_HP) <= 2
    if SHPB == 0  and height <= 6 and VT <= 2 and seg_width <= 6 and cnt >= 2:
        return True

    return False

###############################################################################

def filter_regions(word_img, no_dots_copy, SRL:list, VP:list, upper_base:int, lower_base:int, MTI:int, MFV:int, top_line:int):
    
    valid_separation_regions = []
    overlap = []

    T = 1
    components, labels= cv2.connectedComponents(word_img[:lower_base+5, :], connectivity=8)

    SR_idx = 0
    while SR_idx < len(SRL):
        
        SR = SRL[SR_idx]
        end_idx, cut_idx, start_idx = SR

        # Case 1 : Vertical Projection = 0
        if VP[cut_idx] == 0:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue

      # Case 2 : no connected path between start and end
        # components, labels= cv.connectedComponents(word_img[:, end_idx:start_idx+1], connectivity=8)
        if labels[MTI, end_idx] != labels[MTI, start_idx]:
            valid_separation_regions.append(SR)
            overlap.append(SR)
            SR_idx += 1
            continue

      

        # Case 3 : Contain Holes
        # if check_hole(no_dots_copy[:, end_idx: cut_idx]) and inside_hole(no_dots_copy, end_idx, start_idx):
        cc, l = cv2.connectedComponents(1-(no_dots_copy[:, end_idx:start_idx+1]), connectivity=4)
        
        if cc-1 >= 3 and inside_hole(no_dots_copy, end_idx, start_idx):
            SR_idx += 1
            continue
       
     
        # Case 4 : No baseline between start and end
        segment = no_dots_copy[:, end_idx+1: start_idx]
        segment_width = start_idx-end_idx-1

        j = end_idx+1
        cnt = 0
        while j < start_idx:
            
            # Black pixel (Discontinuity)
            base = upper_base-T
            while base <= lower_base+T:
                
                pixel = no_dots_copy[base][j]
                cnt += pixel

                base += 1
            
            j += 1

        if cnt < segment_width-2 and segment_width > 4:
            
            segment_HP = np.sum(segment, 1).astype('int32')

            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base+T+1:])

            if (int(SHPB) - int(SHPA)) >= 0:
                SR_idx += 1
                continue
            elif VP[cut_idx] <= MFV + T:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
            else:
                SR_idx += 1
                continue

      
        # if SR_idx == 0:
        #     breakpoint()
        # Case 5 : Last region or next VP[nextcut] = 0
        if SR_idx == len(SRL) - 1 or VP[SRL[SR_idx+1][1]] == 0:

            if SR_idx == len(SRL) - 1:
                segment_dots = word_img[:, :SRL[SR_idx][1]+1]
                segment = no_dots_copy[:, :SRL[SR_idx][1]+1]
                next_cut = 0
            else:
                next_cut = SRL[SR_idx+1][1]
                segment_dots = word_img[:, next_cut:SRL[SR_idx][1]+1]
                segment = no_dots_copy[:, next_cut:SRL[SR_idx][1]+1]

            segment_HP = np.sum(segment, 1).astype('int32')
            (h, w) = segment.shape

            top = -1
            for i, proj in enumerate(segment_HP):
                if proj != 0:
                    top = i
                    break
            height = upper_base - top

            # if SR_idx == len(SRL) - 1:
                # breakpoint()
            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base+T+1:])
            sk = skeletonize(segment).astype(np.uint8)
            seg_VP = np.sum(segment, 0).astype('int32')
            non_zero =  np.nonzero(seg_VP)[0]
            cnt = 0
            # for k in range(0, (len(non_zero)//2)+(len(non_zero)%2)):
            for k in range(0, 3):
                if k >= len(non_zero):
                    break
                index = non_zero[k]
                if seg_VP[index] >= height:
                    cnt += 1
            
            if (SHPB <= 5 and cnt > 0 and height <= 6) or (len(non_zero) >= 10 and SHPB > SHPA and not check_dots(segment_dots)):
                SR_idx += 1
                continue
                
        # Strokes 

        SEGP = (-1, -1)
        SEG = (-1, -1)
        SEGN = (-1, -1)
        SEGNN = (-1, -1)
        SEGP_SR1 = (0, 0)
        SEGP_SR2 = (0, 0)
        SEG_SR1 = (0, 0)
        SEG_SR2 = (0, 0)
        SEGN_SR1 = (0, 0)
        SEGN_SR2 = (0, 0)
        SEGNN_SR1 = (0, 0)
        SEGNN_SR2 = (0, 0)

        current_cut = SR[1]
     
        if SR_idx == 0:
            SEGP = (SRL[SR_idx][1], word_img.shape[1]-1)
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx][1], word_img.shape[1]-1)

        if SR_idx > 0:
            SEGP = (SRL[SR_idx][1], SRL[SR_idx-1][1])
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx-1][0], SRL[SR_idx-1][2])
        
        if SR_idx < len(SRL)-1:
            SEG = (SRL[SR_idx+1][1], SRL[SR_idx][1])
            SEG_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEG_SR2 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])

        if SR_idx < len(SRL)-2:
            SEGN = (SRL[SR_idx+2][1], SRL[SR_idx+1][1])
            SEGN_SR1 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])
            SEGN_SR2 = (SRL[SR_idx+2][0], SRL[SR_idx+2][2])
        elif SR_idx == len(SRL)-2:
            SEGN = (0, SRL[SR_idx+1][1])
            SEGN_SR1 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])
            SEGN_SR2 = (0, SRL[SR_idx+1][2])

            
        if SR_idx < len(SRL)-3:
            SEGNN = (SRL[SR_idx+3][1], SRL[SR_idx+2][1])
            SEGNN_SR1 = (SRL[SR_idx+2][0], SRL[SR_idx+2][2])
            SEGNN_SR2 = (SRL[SR_idx+3][0], SRL[SR_idx+3][2])

            
        # if SR_idx == 6:
        #     breakpoint()
        
        # SEG is stroke with dots
        if SEG[0] != -1 and\
            (check_stroke(no_dots_copy, no_dots_copy[:, SEG[0]:SEG[1]], upper_base, lower_base, SEG_SR1, SEG_SR2) \
            and check_dots(word_img[:, SEG[0]:SEG[1]])):
            
            # breakpoint()
            # Case when starts with ش
            if SEGP[0] != -1 and \
                ((check_stroke(no_dots_copy, no_dots_copy[:, SEGP[0]:SEGP[1]], upper_base, lower_base, SEGP_SR1, SEGP_SR2) \
                and not check_dots(word_img[:, SEGP[0]:SEGP[1]]))\
                and (SR_idx == 0 or VP[SRL[SR_idx-1][1]] == 0 or (VP[SRL[SR_idx-1][1]] == 0 and SRL[SR_idx-1] in overlap))):
                
                SR_idx += 2
                continue
            else:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
                
        # SEG is stroke without dots
        elif SEG[0] != -1\
            and (check_stroke(no_dots_copy, no_dots_copy[:, SEG[0]:SEG[1]], upper_base, lower_base, SEG_SR1, SEG_SR2) \
            and not check_dots(word_img[:, SEG[0]:SEG[1]])):

            # Case starts with س
            if SEGP[0] != -1\
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGP[0]:SEGP[1]], upper_base, lower_base, SEGP_SR1, SEGP_SR2) \
                and not check_dots(word_img[:, SEGP[0]:SEGP[1]])):

                SR_idx += 2
                continue

            # SEGN is stroke without dots
            if SEGN[0] != -1 \
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2) \
                and not check_dots(word_img[:, SEGN[0]:SEGN[1]])):

                valid_separation_regions.append(SR)
                SR_idx += 3
                continue

            # SEGN stroke with Dots and SEGNN stroke without Dots
            if SEGN[0] != -1\
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2) \
                and check_dots(word_img[:, SEGN[0]:SEGN[1]])) \
                and ((SEGNN[0] != -1 \
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGNN[0]:SEGNN[1]], upper_base, lower_base, SEGNN_SR1, SEGNN_SR2) \
                and not check_dots(word_img[:, SEGNN[0]:SEGNN[1]]))) or (len(SRL)-1-SR_idx == 2) or (len(SRL)-1-SR_idx == 3)):
        
                    valid_separation_regions.append(SR)
                    SR_idx += 3
                    continue
            
            # SEGN is not stroke or Stroke with Dots
            if SEGN[0] != -1 \
                and ((not check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2)) \
                or (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2) \
                and check_dots(word_img[:, SEGN[0]:SEGN[1]]))):
                    
                    SR_idx += 1
                    continue
            
            SR_idx += 1
            continue
                

        if (len(valid_separation_regions) == 0 or\
            len(valid_separation_regions) > 0 and abs(cut_idx-valid_separation_regions[-1][1]) > 2): 
            valid_separation_regions.append(SR)
        SR_idx += 1

    return valid_separation_regions

###########################################################################

def get_chars(line, word_img):

    # binary_word = binarize(word_img)
    binary_word = word_img//255
    no_dots_copy = remove_dots(binary_word)

    VP_no_dots = np.sum(no_dots_copy, 0).astype('int32')
    VP = np.sum(binary_word, 0).astype('int32')

    # (fill)
    (h, w) = binary_word.shape
    flag = 1
    while flag:
        flag = 0
        for row in range(h-1):
            for col in range(1, w-1):

                if binary_word[row][col] == 0 and binary_word[row][col-1] == 1 and binary_word[row][col+1] == 1 and binary_word[row+1][col] == 1 and VP_no_dots[col] != 0:
                    binary_word[row][col] = 1

    # (baseline_detection)
    # Get baseline index of a given word
    HP = np.sum(remove_dots(line), 1).astype('int32')
    peak = np.amax(HP)

    # Array of indices of max element
    baseline_idx = np.where(HP == peak)[0]

    # Get first or last index
    upper_base = baseline_idx[0]
    lower_base = baseline_idx[-1]
    thickness = abs(lower_base - upper_base) + 1


    no_dots_copy = remove_dots(binary_word)

    MTI = horizontal_transitions(no_dots_copy, upper_base)
        
    SRL, wrong = cut_points(binary_word, VP, thickness, MTI, upper_base)

    if wrong:
        MTI -= 1
        SRL.clear()
        SRL, wrong = cut_points(binary_word, VP, thickness, MTI, upper_base)

    top_line = -1

    valid = filter_regions(binary_word, no_dots_copy, SRL, VP, upper_base, lower_base, MTI, thickness, top_line)

    # (extract_char)
    # binary image needs to be (0, 255) to be saved on disk not (0, 1)
    binary_word = binary_word * 255
    h, w = binary_word.shape

    next_cut = w
    char_imgs = []

    for SR in valid:
        char_imgs.append(binary_word[:, SR[1]:next_cut])
        next_cut = SR[1]
    char_imgs.append(binary_word[:, 0:next_cut])

    return char_imgs


###############################################################################
###############################################################################
###############################################################################

def bound_box(img_char):
    HP = np.sum(img_char, 1).astype('int32')
    VP = np.sum(img_char, 0).astype('int32')
    #cumsum
    top = -1
    down = -1
    left = -1
    right = -1

    i = 0
    while i < len(HP):
        if HP[i] != 0:
            top = i
            break
        i += 1

    i = len(HP)-1
    while i >= 0:
        if HP[i] != 0:
            down = i
            break
        i -= 1

    i = 0
    while i < len(VP):
        if VP[i] != 0:
            left = i
            break
        i += 1

    i = len(VP)-1
    while i >= 0:
        if VP[i] != 0:
            right = i
            break
        i -= 1

    return img_char[top:down+1, left:right+1]