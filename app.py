# =============================
# 🎓 ABOUT & CREDITS FOOTER
# =============================

st.markdown("""
<style>
.custom-footer-wrapper {
    display: flex;
    justify-content: center;
    margin-top: 4rem;
}
.custom-footer-box {
    background: rgba(0, 0, 0, 0.6);  /* Background behind content */
    padding: 30px;
    border-radius: 15px;
    border: 1px solid #00c6ff;
    width: 100%;
    max-width: 1000px;
}
.custom-footer-box h4 {
    color: #fff;
    margin-bottom: 10px;
}
.custom-footer-box p, .custom-footer-box a {
    color: #ccc;
    font-size: 16px;
}
.custom-footer-box a {
    color: #00c6ff;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-footer-wrapper"><div class="custom-footer-box">', unsafe_allow_html=True)

logo_url = "https://tse2.mm.bing.net/th/id/OIP.WC5xs7MJrmfk_YEHDn6BOAAAAA?pid=Api&P=0&h=180"

f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    st.markdown(f'<div style="text-align: center;"><img src="{logo_url}" width="150"></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Cadi Ayyad University<br>Faculties of Medicine and Pharmacie</p>", unsafe_allow_html=True)
with f_col2:
    st.markdown(
        """
        <div>
            <h4>Developed By</h4>
            <p>ELLAOUAH ZAHIRA | <a href="mailto:zahiraellaouah@gmail.com">📧 zahiraellaouah@gmail.com</a></p>
            <h4>Under the Supervision of</h4>
            <p>Pr. Nezha Oumghar &nbsp;&nbsp;&nbsp;&nbsp; Pr. Mohamed Amine Chadi</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div></div>', unsafe_allow_html=True)
