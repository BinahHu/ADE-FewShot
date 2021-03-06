import torch
import torch.nn as nn
from model.component.classifier import Classifier, CosClassifier
from model.component.resnet import resnet18, resnet34, resnet10
from model.component.attr import AttrClassifier
from model.component.scene import SceneClassifier
from model.component.seg import MaskPredictor
from model.component.bbox import BBoxModule
from model.component.bkg import FullMaskPredictor
from model.component.hierarchy import HierarchyClassifier
from model.component.part import PartClassifier
from model.component.patch_location import PatchLocationClassifier
from model.component.rotation import RotationClassifier


class ModelBuilder:
    # weight initialization
    def __init__(self, args):
        self.args = args

    def weight_init(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif class_name.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.01)

    def build_backbone(self):
        if self.args.architecture == 'resnet18':
            backbone = resnet18()
        elif self.args.architecture == 'resnet10':
            backbone = resnet10()
        elif self.args.architecture == 'resnet34':
            backbone = resnet34()

        backbone.apply(self.weight_init)
        return backbone

    def build_classifier(self):
        if self.args.cls == 'Linear':
            classifier = Classifier(self.args)
        elif self.args.cls == 'Cos':
            classifier = CosClassifier(self.args)
        classifier.apply(self.weight_init)
        return classifier

    def build_attr(self):
        attr_classifier = AttrClassifier(self.args)
        attr_classifier.apply(self.weight_init)
        return attr_classifier

    def build_part(self):
        part_classifier = PartClassifier(self.args)
        part_classifier.apply(self.weight_init)
        return part_classifier

    def build_scene(self):
        scene_classifier = SceneClassifier(self.args)
        scene_classifier.apply(self.weight_init)
        return scene_classifier

    def build_seg(self):
        segment_module = MaskPredictor(self.args)
        segment_module.apply(self.weight_init)
        return segment_module

    def build_bkg(self):
        background_module = FullMaskPredictor(self.args)
        background_module.apply(self.weight_init)
        return background_module

    def build_bbox(self):
        bbox_module = BBoxModule(self.args)
        bbox_module.apply(self.weight_init)
        return bbox_module

    def build_hierarchy(self):
        hierarchy_classifier = HierarchyClassifier(self.args)
        hierarchy_classifier.apply(self.weight_init)
        return hierarchy_classifier

    def build_patch_location(self):
        patch_location_classifier = PatchLocationClassifier(self.args)
        patch_location_classifier.apply(self.weight_init)
        return patch_location_classifier

    def build_rotation(self):
        rotation_classifier = RotationClassifier(self.args)
        rotation_classifier.apply(self.weight_init)
        return rotation_classifier
