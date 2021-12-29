SQuAD数据集
    斯坦福的问答数据集，由众包人员基于一系列维基百科文章的提问和对应的答案构成，其中每个问题的答案是相关文章中的文本片段或区间。SQuAD 包含关于 500 多篇文章的超过 100000 个问答对，

SQuAD数据集特点：
i) SQuAD 是一个封闭的数据集，这意味着问题的答案通常位于文章的某一个区间中。
ii) 寻找答案的过程可以简化为在文中找到与答案相对应部分的起始索引和结束索引。
iii) 75% 的答案长度小于四个单词。

示例：
    语境：
     The content of the acts, particularly section 1 (1) of the amending act of 1938, shows the importance which was then attached to giving architects the responsibility of superintending or supervising the building works of local authorities (for housing and other projects), rather than persons professionally qualified only as municipal or other engineers. By the 1970s another issue had emerged affecting education for qualification and registration for practice as an architect, due to the obligation imposed on the United Kingdom and other European governments to comply with European Union Directives concerning mutual recognition of professional qualifications in favour of equal standards across borders, in furtherance of the policy for a single market of the European Union. This led to proposals for reconstituting ARCUK. Eventually, in the 1990s, before proceeding, the government issued a consultation paper "Reform of Architects Registration" (1994). The change of name to "Architects Registration Board" was one of the proposals which was later enacted in the Housing Grants, Construction and Regeneration Act 1996 and reenacted as the Architects Act 1997; another was the abolition of the ARCUK Board of Architectural Education.
    问题：
    What was the new name given to ARCUK in the '90s?
    答案：
    {'text': ['Architects Registration Board'], 'answer_start': [985]}
    

 

