class MnistLabelParser:
    def __init__(self):
        self.location_dict = {
            'center': 0,
            'upper left': 1,
            'upper right': 2,
            'lower left': 3,
            'lower right': 4,
            'else': -100,
        }
        self.color_dict = {
            'white': 0,
            'red': 1,
            'green': 2,
            'blue': 3,
            'else': -101,
        }
        self.center_total_count = 0
        self.center_correct_count = 0
        self.quad1_total_count = 0
        self.quad1_correct_count = 0
        self.quad2_total_count = 0
        self.quad2_correct_count = 0
        self.quad3_total_count = 0
        self.quad3_correct_count = 0
        self.quad4_total_count = 0
        self.quad4_correct_count = 0

        self.center_color_correct_count = 0
        self.center_number_correct_count = 0

        self.quad1_color_correct_count = 0
        self.quad1_number_correct_count = 0

        self.quad2_color_correct_count = 0
        self.quad2_number_correct_count = 0

        self.quad3_color_correct_count = 0
        self.quad3_number_correct_count = 0

        self.quad4_color_correct_count = 0
        self.quad4_number_correct_count = 0


    def get_location_index(self, t):
        """
        input: text unit(str)
        output: label(int)
        """
        if ('upper left' in t) or ('top left' in t):
            return self.location_dict['upper left']
        elif ('upper right' in t) or ('top right' in t):
            return self.location_dict['upper right']
        elif ('lower left' in t) or ('bottom left' in t):
            return self.location_dict['lower left']
        elif ('lower right'in t) or ('bottom right' in t):
            return self.location_dict['lower right']
        else:
            return self.location_dict['center']

    def get_color_index(self, t):
        """
        input: text unit(str)
        output: label(int)
        """
        if 'white' in t:
            return self.color_dict['white']
        elif 'red' in t:
            return self.color_dict['red']
        elif 'green' in t:
            return self.color_dict['green']
        elif 'blue' in t:
            return self.color_dict['blue']
        else:
            return self.color_dict['else']

    def get_number_index(self, t):
        """
        input: text unit(str)
        output: label(int)
        """
        for i in range(10):
            if str(i) in t:
                return int(i)
        return -102

    def parse_label(self, text):
        """
        input: text(str)
        output: labels(list of lists)
        """
        # first, split the text
        text_unit_list = text.split(',')
        results = list()
        for t in text_unit_list:
            result = list()
            # append location idx
            result.append(self.get_location_index(t))
            # append color idx
            result.append(self.get_color_index(t))
            # append number idx
            result.append(self.get_number_index(t))
            results.append(result)
        
        return results # eg. [[2, 1, 4], [1, 0, 1], [4, 3, 9]]

    def count_correct(self, gt_text, gen_text):
        count = 0
        color_count = 0
        number_count = 0
        gt_labels_list = self.parse_label(gt_text)
        gen_labels_list = self.parse_label(gen_text)
        num_labels = len(gt_labels_list)
        gt_labels_loc_indices = [label[0] for label in gt_labels_list]
        
        for label in gen_labels_list:
            #
            if label in gt_labels_list:
                count += 1        
            #
            if label[0] in gt_labels_loc_indices:
                idx = gt_labels_loc_indices.index(label[0])
                if label[1] == gt_labels_list[idx][1]:
                    color_count += 1
                if label[2] == gt_labels_list[idx][2]:
                    number_count += 1

        if num_labels == 4:
            self.quad4_total_count += num_labels
            self.quad4_correct_count += count
            self.quad4_color_correct_count += color_count
            self.quad4_number_correct_count += number_count
        elif num_labels == 3:
            self.quad3_total_count += num_labels
            self.quad3_correct_count += count
            self.quad3_color_correct_count += color_count
            self.quad3_number_correct_count += number_count
        elif num_labels == 2:
            self.quad2_total_count += num_labels
            self.quad2_correct_count += count
            self.quad2_color_correct_count += color_count
            self.quad2_number_correct_count += number_count
        else: # num_labels == 1
            if gt_labels_list[0][0] == 0: # location == 0(center)
                self.center_total_count += num_labels
                self.center_correct_count += count
                self.center_color_correct_count += color_count
                self.center_number_correct_count += number_count
            else:
                self.quad1_total_count += num_labels
                self.quad1_correct_count += count
                self.quad1_color_correct_count += color_count
                self.quad1_number_correct_count += number_count

    def calc_accuracy(self, eps=1e-8):
        center_acc = self.center_correct_count / (self.center_total_count + eps)
        quad1_acc = self.quad1_correct_count / (self.quad1_total_count + eps)
        quad2_acc = self.quad2_correct_count / (self.quad2_total_count + eps)
        quad3_acc = self.quad3_correct_count / (self.quad3_total_count + eps)
        quad4_acc = self.quad4_correct_count / (self.quad4_total_count + eps)

        center_color_acc = self.center_color_correct_count / (self.center_total_count + eps)
        center_number_acc = self.center_number_correct_count / (self.center_total_count + eps)
        quad1_color_acc = self.quad1_color_correct_count / (self.quad1_total_count + eps)
        quad1_number_acc = self.quad1_number_correct_count / (self.quad1_total_count + eps)
        quad2_color_acc = self.quad2_color_correct_count / (self.quad2_total_count + eps)
        quad2_number_acc = self.quad2_number_correct_count / (self.quad2_total_count + eps)
        quad3_color_acc = self.quad3_color_correct_count / (self.quad3_total_count + eps)
        quad3_number_acc = self.quad3_number_correct_count / (self.quad3_total_count + eps)
        quad4_color_acc = self.quad4_color_correct_count / (self.quad4_total_count + eps)
        quad4_number_acc = self.quad4_number_correct_count / (self.quad4_total_count + eps)

        return [center_acc, quad1_acc, quad2_acc, quad3_acc, quad4_acc], \
            [center_color_acc, center_number_acc, quad1_color_acc, quad1_number_acc, quad2_color_acc, quad2_number_acc, quad3_color_acc, quad3_number_acc, quad4_color_acc, quad4_number_acc]



class ImgAcc:
    def __init__(self):
        self.center_total_count = 0
        self.center_correct_count = 0
        self.center_color_correct_count = 0
        self.center_number_correct_count = 0

        self.quad1_total_count = 0
        self.quad1_correct_count = 0
        self.quad1_color_correct_count = 0
        self.quad1_number_correct_count = 0

        self.quad2_total_count = 0
        self.quad2_correct_count = 0
        self.quad2_color_correct_count = 0
        self.quad2_number_correct_count = 0

        self.quad3_total_count = 0
        self.quad3_correct_count = 0
        self.quad3_color_correct_count = 0
        self.quad3_number_correct_count = 0

        self.quad4_total_count = 0
        self.quad4_correct_count = 0
        self.quad4_color_correct_count = 0
        self.quad4_number_correct_count = 0

    def count_correct(self, gt_labels_list, gen_labels_list): # eg. [[1,0,0], [2,1,3]]    [[1,0,1],[2,1,4]]
        count = 0
        color_count = 0
        number_count = 0
        num_labels = len(gt_labels_list)
        gt_labels_loc_indices = [label[0] for label in gt_labels_list]
        
        for label in gen_labels_list:
            #
            if label in gt_labels_list:
                count += 1        
            #
            if label[0] in gt_labels_loc_indices:
                idx = gt_labels_loc_indices.index(label[0])
                if label[1] == gt_labels_list[idx][1]:
                    color_count += 1
                if label[2] == gt_labels_list[idx][2]:
                    number_count += 1

        if num_labels == 4:
            self.quad4_total_count += num_labels
            self.quad4_correct_count += count
            self.quad4_color_correct_count += color_count
            self.quad4_number_correct_count += number_count
        elif num_labels == 3:
            self.quad3_total_count += num_labels
            self.quad3_correct_count += count
            self.quad3_color_correct_count += color_count
            self.quad3_number_correct_count += number_count
        elif num_labels == 2:
            self.quad2_total_count += num_labels
            self.quad2_correct_count += count
            self.quad2_color_correct_count += color_count
            self.quad2_number_correct_count += number_count
        else: # num_labels == 1
            if gt_labels_list[0][0] == 0: # location == 0(center)
                self.center_total_count += num_labels
                self.center_correct_count += count
                self.center_color_correct_count += color_count
                self.center_number_correct_count += number_count
            else:
                self.quad1_total_count += num_labels
                self.quad1_correct_count += count
                self.quad1_color_correct_count += color_count
                self.quad1_number_correct_count += number_count
        
    def calc_accuracy(self, eps=1e-8):
        center_acc = self.center_correct_count / (self.center_total_count + eps)
        quad1_acc = self.quad1_correct_count / (self.quad1_total_count + eps)
        quad2_acc = self.quad2_correct_count / (self.quad2_total_count + eps)
        quad3_acc = self.quad3_correct_count / (self.quad3_total_count + eps)
        quad4_acc = self.quad4_correct_count / (self.quad4_total_count + eps)

        center_color_acc = self.center_color_correct_count / (self.center_total_count + eps)
        center_number_acc = self.center_number_correct_count / (self.center_total_count + eps)
        quad1_color_acc = self.quad1_color_correct_count / (self.quad1_total_count + eps)
        quad1_number_acc = self.quad1_number_correct_count / (self.quad1_total_count + eps)
        quad2_color_acc = self.quad2_color_correct_count / (self.quad2_total_count + eps)
        quad2_number_acc = self.quad2_number_correct_count / (self.quad2_total_count + eps)
        quad3_color_acc = self.quad3_color_correct_count / (self.quad3_total_count + eps)
        quad3_number_acc = self.quad3_number_correct_count / (self.quad3_total_count + eps)
        quad4_color_acc = self.quad4_color_correct_count / (self.quad4_total_count + eps)
        quad4_number_acc = self.quad4_number_correct_count / (self.quad4_total_count + eps)

        return [center_acc, quad1_acc, quad2_acc, quad3_acc, quad4_acc], \
            [center_color_acc, center_number_acc, quad1_color_acc, quad1_number_acc, quad2_color_acc, quad2_number_acc, quad3_color_acc, quad3_number_acc, quad4_color_acc, quad4_number_acc]


class FashionLabelParser:
    def __init__(self):
        self.location_dict = {
            'center': 0,
            'upper left': 1,
            'upper right': 2,
            'lower left': 3,
            'lower right': 4,
            'else': -100,
        }
        self.color_dict = {
            'white': 0,
            'red': 1,
            'green': 2,
            'blue': 3,
            'else': -101,
        }
        self.item_dict = {
            'tshirt': 0,
            'trouser': 1,
            'pullover': 2,
            'dress': 3,
            'coat': 4,
            'sandal': 5,
            'shirt': 6,
            'sneaker': 7,
            'bag': 8,
            'ankle boot': 9,
            'else': -102,
        }
        self.center_total_count = 0
        self.center_correct_count = 0
        self.quad1_total_count = 0
        self.quad1_correct_count = 0
        self.quad2_total_count = 0
        self.quad2_correct_count = 0
        self.quad3_total_count = 0
        self.quad3_correct_count = 0
        self.quad4_total_count = 0
        self.quad4_correct_count = 0

        self.center_color_correct_count = 0
        self.center_number_correct_count = 0

        self.quad1_color_correct_count = 0
        self.quad1_number_correct_count = 0

        self.quad2_color_correct_count = 0
        self.quad2_number_correct_count = 0

        self.quad3_color_correct_count = 0
        self.quad3_number_correct_count = 0

        self.quad4_color_correct_count = 0
        self.quad4_number_correct_count = 0


    def get_location_index(self, t):
        """
        input: text unit(str)
        output: label(int)
        """
        if ('upper left' in t) or ('top left' in t):
            return self.location_dict['upper left']
        elif ('upper right' in t) or ('top right' in t):
            return self.location_dict['upper right']
        elif ('lower left' in t) or ('bottom left' in t):
            return self.location_dict['lower left']
        elif ('lower right'in t) or ('bottom right' in t):
            return self.location_dict['lower right']
        else:
            return self.location_dict['center']

    def get_color_index(self, t):
        """
        input: text unit(str)
        output: label(int)
        """
        if 'white' in t:
            return self.color_dict['white']
        elif 'red' in t:
            return self.color_dict['red']
        elif 'green' in t:
            return self.color_dict['green']
        elif 'blue' in t:
            return self.color_dict['blue']
        else:
            return self.color_dict['else']

    def get_item_index(self, t):
        """
        input: text unit(str)
        output: label(int)
        """
        if 'tshirt' in t:
            return self.item_dict['tshirt']
        elif 'trouser' in t:
            return self.item_dict['trouser']
        elif 'pullover' in t:
            return self.item_dict['pullover']
        elif 'dress' in t:
            return self.item_dict['dress']
        elif 'coat' in t:
            return self.item_dict['coat']
        elif 'sandal' in t:
            return self.item_dict['sandal']
        elif ' shirt' in t:   # NOTE: not 'shirt'
            return self.item_dict['shirt']
        elif 'sneaker' in t:
            return self.item_dict['sneaker']
        elif 'bag' in t:
            return self.item_dict['bag']
        elif 'ankle boot' in t:
            return self.item_dict['ankle boot']
        else:
            return self.color_dict['else']

    def parse_label(self, text):
        """
        input: text(str)
        output: labels(list of lists)
        """
        # first of all, split the text
        text_unit_list = text.split(',')
        results = list()
        for t in text_unit_list:
            result = list()
            # append location idx
            result.append(self.get_location_index(t))
            # append color idx
            result.append(self.get_color_index(t))
            # append number idx
            result.append(self.get_item_index(t))
            results.append(result)
        
        return results # eg. [[2, 1, 4], [1, 0, 1], [4, 3, 9]]

    def count_correct(self, gt_text, gen_text):
        count = 0
        color_count = 0
        number_count = 0
        gt_labels_list = self.parse_label(gt_text)
        gen_labels_list = self.parse_label(gen_text)
        num_labels = len(gt_labels_list)
        gt_labels_loc_indices = [label[0] for label in gt_labels_list]
        
        for label in gen_labels_list:
            #
            if label in gt_labels_list:
                count += 1        
            #
            if label[0] in gt_labels_loc_indices:
                idx = gt_labels_loc_indices.index(label[0])
                if label[1] == gt_labels_list[idx][1]:
                    color_count += 1
                if label[2] == gt_labels_list[idx][2]:
                    number_count += 1

        if num_labels == 4:
            self.quad4_total_count += num_labels
            self.quad4_correct_count += count
            self.quad4_color_correct_count += color_count
            self.quad4_number_correct_count += number_count
        elif num_labels == 3:
            self.quad3_total_count += num_labels
            self.quad3_correct_count += count
            self.quad3_color_correct_count += color_count
            self.quad3_number_correct_count += number_count
        elif num_labels == 2:
            self.quad2_total_count += num_labels
            self.quad2_correct_count += count
            self.quad2_color_correct_count += color_count
            self.quad2_number_correct_count += number_count
        else: # num_labels == 1
            if gt_labels_list[0][0] == 0: # location == 0(center)
                self.center_total_count += num_labels
                self.center_correct_count += count
                self.center_color_correct_count += color_count
                self.center_number_correct_count += number_count
            else:
                self.quad1_total_count += num_labels
                self.quad1_correct_count += count
                self.quad1_color_correct_count += color_count
                self.quad1_number_correct_count += number_count

    def calc_accuracy(self, eps=1e-8):
        center_acc = self.center_correct_count / (self.center_total_count + eps)
        quad1_acc = self.quad1_correct_count / (self.quad1_total_count + eps)
        quad2_acc = self.quad2_correct_count / (self.quad2_total_count + eps)
        quad3_acc = self.quad3_correct_count / (self.quad3_total_count + eps)
        quad4_acc = self.quad4_correct_count / (self.quad4_total_count + eps)

        center_color_acc = self.center_color_correct_count / (self.center_total_count + eps)
        center_number_acc = self.center_number_correct_count / (self.center_total_count + eps)
        quad1_color_acc = self.quad1_color_correct_count / (self.quad1_total_count + eps)
        quad1_number_acc = self.quad1_number_correct_count / (self.quad1_total_count + eps)
        quad2_color_acc = self.quad2_color_correct_count / (self.quad2_total_count + eps)
        quad2_number_acc = self.quad2_number_correct_count / (self.quad2_total_count + eps)
        quad3_color_acc = self.quad3_color_correct_count / (self.quad3_total_count + eps)
        quad3_number_acc = self.quad3_number_correct_count / (self.quad3_total_count + eps)
        quad4_color_acc = self.quad4_color_correct_count / (self.quad4_total_count + eps)
        quad4_number_acc = self.quad4_number_correct_count / (self.quad4_total_count + eps)

        return [center_acc, quad1_acc, quad2_acc, quad3_acc, quad4_acc], \
            [center_color_acc, center_number_acc, quad1_color_acc, quad1_number_acc, quad2_color_acc, quad2_number_acc, quad3_color_acc, quad3_number_acc, quad4_color_acc, quad4_number_acc]
