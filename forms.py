from django import forms


class dato_RedOb(forms.Form):
    #txtObjeto = forms.CharField(widget=forms.Textarea)
    txtObjeto = forms.Select()


class dato_Tag(forms.Form):
    txtTag = forms.CharField(widget=forms.Textarea)
